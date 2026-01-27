from typing import Dict

import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from editor.base import BaseEditor
from util import get_module, get_shape


class HORSE(BaseEditor):
    
    def _load_cached_tensors(self, module_idx: int) -> tuple:
        n_batches = math.ceil(self.config.data.n_edits / self.config.data.batch_size)
        
        keys = torch.cat([
            torch.load(f"{self.config.editor.cache_dir}/{module_idx}_{idx}_keys.pth")
            for idx in range(n_batches)
        ])
        values_grad = torch.cat([
            torch.load(f"{self.config.editor.cache_dir}/{module_idx}_{idx}_values_grad.pth")
            for idx in range(n_batches)
        ])
        return keys, values_grad
    
    def _compute_value_diffs(self, net, keys, values_grad, layer_idx) -> torch.Tensor:
        batch_size = self.config.editor.batch_size
        n_samples = keys.shape[0]
        n_batches = math.ceil(n_samples / batch_size)
        
        value_diffs = torch.zeros(
            (n_samples, net.value_size), 
            device=self.config.editor_device
        )
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            pesudo_keys, pesudo_values_grad = net(
                keys[start_idx:end_idx],
                values_grad[start_idx:end_idx],
                layer_idx
            )
            coeffs = -net.lr(layer_idx) * (keys[start_idx:end_idx] * pesudo_keys).sum(-1).unsqueeze(-1)
            value_diffs[start_idx:end_idx] = coeffs * pesudo_values_grad
            
        return value_diffs
    
    def _compute_mat(self, net, keys, layer_idx) -> torch.Tensor:
        return keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(
            net.key_size, device=self.config.editor_device
        )
    
    def _apply_orthogonal_projection(self, value_diffs, value_diffs_last, values_grad_last) -> torch.Tensor:
        projection = torch.dot(
            value_diffs.view(-1), 
            value_diffs_last.view(-1)
        ) / torch.dot(
            value_diffs_last.view(-1), 
            value_diffs_last.view(-1)
        )
        return value_diffs - projection * values_grad_last

    def predict_param_shifts(self) -> Dict[str, torch.FloatTensor]:
        param_shifts = {}
        cache_dir = self.config.editor.cache_dir
        
        for module_idx, module_name in enumerate(self.config.model.edit_modules):
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            
            keys, values_grad = self._load_cached_tensors(module_idx)
            
            with torch.no_grad():
                value_diffs = self._compute_value_diffs(net, keys, values_grad, layer_idx)
                mat = self._compute_mat(net, keys, layer_idx)
            
            torch.save(value_diffs, f"{cache_dir}/{module_idx}_value_diffs.pth")
            torch.save(values_grad, f"{cache_dir}/{module_idx}_values_grad.pth")
            
            if module_idx != 0:
                value_diffs_last = torch.load(f"{cache_dir}/{module_idx - 1}_value_diffs.pth")
                values_grad_last = torch.load(f"{cache_dir}/{module_idx - 1}_values_grad.pth")
                value_diffs = self._apply_orthogonal_projection(value_diffs, value_diffs_last, values_grad_last)
            
            param_shift = torch.linalg.solve(mat, keys.T @ value_diffs)
            param_shifts[module_name] = param_shift.to(next(self.model.parameters()).device)

        return param_shifts

    def update_hypernet(self, param_shifts: Dict[str, torch.FloatTensor]):
        self.opt.zero_grad()
        cache_dir = self.config.editor.cache_dir
        
        for module_idx, module_name in enumerate(self.config.model.edit_modules):
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            
            keys, values_grad = self._load_cached_tensors(module_idx)
            
            module = get_module(self.model, module_name)
            module_grad = module.weight.grad.to(torch.float32).to(self.config.editor_device)
            if isinstance(module, nn.Linear):
                module_grad = module_grad.T
            
            with torch.no_grad():
                mat = torch.linalg.solve(
                    keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device=self.config.editor_device),
                    module_grad
                )
            value_diffs_grad = keys @ mat
            
            value_diffs = self._compute_value_diffs_with_grad(net, keys, values_grad, layer_idx)
            
            torch.save(value_diffs.detach(), f"{cache_dir}/{module_idx}_value_diffs_hyp.pth")
            torch.save(values_grad, f"{cache_dir}/{module_idx}_values_grad_hyp.pth")
            
            if module_idx != 0:
                value_diffs_last = torch.load(f"{cache_dir}/{module_idx - 1}_value_diffs_hyp.pth")
                values_grad_last = torch.load(f"{cache_dir}/{module_idx - 1}_values_grad_hyp.pth")
                projection = torch.dot(
                    value_diffs.view(-1), 
                    value_diffs_last.view(-1)
                ) / torch.dot(
                    value_diffs_last.view(-1), 
                    value_diffs_last.view(-1)
                )
                value_diffs = value_diffs - projection * values_grad_last
            
            (value_diffs_grad * value_diffs).sum().backward()
        
        clip_grad_norm_(self.net.parameters(), self.config.editor.max_grad_norm)
        self.opt.step()
    
    def _compute_value_diffs_with_grad(self, net, keys, values_grad, layer_idx) -> torch.Tensor:
        batch_size = self.config.editor.batch_size
        n_samples = keys.shape[0]
        
        value_diffs_list = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            pesudo_keys, pesudo_values_grad = net(
                keys[start_idx:end_idx],
                values_grad[start_idx:end_idx],
                layer_idx
            )
            coeffs = -net.lr(layer_idx) * (keys[start_idx:end_idx] * pesudo_keys).sum(-1).unsqueeze(-1)
            value_diffs_list.append(coeffs * pesudo_values_grad)
        
        return torch.cat(value_diffs_list, dim=0)
