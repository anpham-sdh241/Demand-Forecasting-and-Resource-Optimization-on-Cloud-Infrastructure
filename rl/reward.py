"""
Reward Function for VM Allocation Environment
==============================================

Implements reward computation for two scenarios:
1. Minimize Resource Overload
2. Optimize Operational Cost

Reward formula:
    reward = - α * sla_violation
             - β * overflow_cpu_ram
             - γ * switching_cost
             - δ * vm_cost
             + ε * efficiency_bonus
"""

from typing import Dict, Tuple
import numpy as np
from .config import RewardConfig


def compute_sla_violation(
    cpu_required: float,
    mem_required: float,
    cpu_threshold: float,
    mem_threshold: float,
    cpu_allocated: float,
    mem_allocated: float,
    config: RewardConfig,
) -> Tuple[float, bool]:
    """
    Compute SLA violation penalty.
    
    SLA violation occurs when total available (host + VMs) < required resources.
    Host handles demand up to threshold, VMs handle the overflow.
    
    Args:
        cpu_required: Required CPU cores (total demand)
        mem_required: Required memory in GB (total demand)
        cpu_threshold: Host CPU threshold (what host can handle)
        mem_threshold: Host memory threshold (what host can handle)
        cpu_allocated: Allocated CPU cores from VMs
        mem_allocated: Allocated memory in GB from VMs
        config: Reward configuration
        
    Returns:
        Tuple of (penalty, is_violated)
    """
    # Total available = Host threshold + VM resources
    total_cpu_available = cpu_threshold + cpu_allocated
    total_mem_available = mem_threshold + mem_allocated
    
    # Shortage = demand - total available
    cpu_shortage = max(0, cpu_required - total_cpu_available)
    mem_shortage = max(0, mem_required - total_mem_available)
    
    is_violated = cpu_shortage > 0 or mem_shortage > 0
    
    penalty = (
        cpu_shortage * config.sla_penalty_per_core +
        mem_shortage * config.sla_penalty_per_gb
    )
    
    return penalty, is_violated


def compute_overflow_penalty(
    cpu_required: float,
    mem_required: float,
    cpu_threshold: float,
    mem_threshold: float,
    config: RewardConfig,
) -> float:
    """
    Compute resource overflow penalty (demand exceeding host capacity).
    
    Args:
        cpu_required: Required CPU cores
        mem_required: Required memory in GB
        cpu_threshold: CPU threshold (cores)
        mem_threshold: Memory threshold (GB)
        config: Reward configuration
        
    Returns:
        Overflow penalty value
    """
    cpu_overflow = max(0, cpu_required - cpu_threshold)
    mem_overflow = max(0, mem_required - mem_threshold)
    
    penalty = (
        cpu_overflow * config.overflow_penalty_per_core +
        mem_overflow * config.overflow_penalty_per_gb
    )
    
    return penalty


def compute_switching_cost(
    prev_vms: Dict[str, int],
    curr_vms: Dict[str, int],
    vm_catalog: Dict[str, Dict],
) -> float:
    """
    Compute cost of switching VMs (starting/stopping).
    
    Args:
        prev_vms: Previous VM allocation {vm_type: count}
        curr_vms: Current VM allocation {vm_type: count}
        vm_catalog: VM specifications with switching_cost
        
    Returns:
        Total switching cost
    """
    total_switching_cost = 0.0
    
    all_vm_types = set(prev_vms.keys()) | set(curr_vms.keys())
    
    for vm_type in all_vm_types:
        prev_count = prev_vms.get(vm_type, 0)
        curr_count = curr_vms.get(vm_type, 0)
        
        # Number of VMs changed (both starts and stops incur cost)
        changes = abs(curr_count - prev_count)
        
        if changes > 0 and vm_type in vm_catalog:
            switching_cost = vm_catalog[vm_type].get("switching_cost", 0.0)
            total_switching_cost += changes * switching_cost
    
    return total_switching_cost


def compute_vm_cost(
    vm_allocation: Dict[str, int],
    vm_catalog: Dict[str, Dict],
) -> float:
    """
    Compute hourly VM operational cost.
    
    Args:
        vm_allocation: VM allocation {vm_type: count}
        vm_catalog: VM specifications with cost_per_hour
        
    Returns:
        Total cost per hour
    """
    total_cost = 0.0
    
    for vm_type, count in vm_allocation.items():
        if vm_type in vm_catalog and count > 0:
            cost_per_hour = vm_catalog[vm_type].get("cost_per_hour", 0.0)
            total_cost += count * cost_per_hour
    
    return total_cost


def compute_efficiency_bonus(
    cpu_overflow: float,
    mem_overflow: float,
    cpu_allocated: float,
    mem_allocated: float,
) -> float:
    """
    Compute efficiency bonus for good VM resource utilization.
    
    Bonus is given when VM utilization is high (80-95%) without shortage.
    Efficiency is based on how well VMs cover the overflow (demand beyond host threshold).
    
    Args:
        cpu_overflow: CPU overflow that VMs need to handle (demand - host_threshold)
        mem_overflow: Memory overflow that VMs need to handle (demand - host_threshold)
        cpu_allocated: Allocated CPU cores from VMs
        mem_allocated: Allocated memory in GB from VMs
        
    Returns:
        Efficiency bonus (-1.0 to 1.0)
    """
    # If no overflow, no VMs needed - give bonus for not over-provisioning
    if cpu_overflow <= 0 and mem_overflow <= 0:
        if cpu_allocated <= 0 and mem_allocated <= 0:
            return 1.0  # Perfect: no overflow, no VMs
        else:
            # Penalty proportional to over-provision amount
            waste = cpu_allocated + mem_allocated
            return -0.5 * min(waste, 2.0)  # Cap penalty at -1.0
    
    # If overflow but no VMs allocated, penalty
    if cpu_allocated <= 0 and mem_allocated <= 0:
        return -1.0  # Bad: overflow but no VMs
    
    # Calculate utilization percentages (how well VMs cover the overflow)
    cpu_util = (cpu_overflow / cpu_allocated * 100) if cpu_allocated > 0 else 0
    mem_util = (mem_overflow / mem_allocated * 100) if mem_allocated > 0 else 0
    
    # Bonus for high utilization (70-100% is good, 80-95% is optimal)
    cpu_bonus = 0.0
    mem_bonus = 0.0
    
    if 80 <= cpu_util <= 95:
        cpu_bonus = 1.0
    elif 70 <= cpu_util < 80 or 95 < cpu_util <= 100:
        cpu_bonus = 0.7
    elif 50 <= cpu_util < 70:
        cpu_bonus = 0.3
    elif cpu_util > 100:
        cpu_bonus = -0.5  # Penalty for under-provisioning (shortage)
    elif cpu_util < 50:
        # Stronger penalty for heavy over-provisioning
        cpu_bonus = -0.5 * (1 - cpu_util / 50)  # -0.5 at 0%, -0.25 at 25%
    
    if 80 <= mem_util <= 95:
        mem_bonus = 1.0
    elif 70 <= mem_util < 80 or 95 < mem_util <= 100:
        mem_bonus = 0.7
    elif 50 <= mem_util < 70:
        mem_bonus = 0.3
    elif mem_util > 100:
        mem_bonus = -0.5  # Penalty for under-provisioning (shortage)
    elif mem_util < 50:
        # Stronger penalty for heavy over-provisioning
        mem_bonus = -0.5 * (1 - mem_util / 50)
    
    return (cpu_bonus + mem_bonus) / 2


def compute_reward(
    cpu_required: float,
    mem_required: float,
    cpu_threshold: float,
    mem_threshold: float,
    cpu_allocated: float,
    mem_allocated: float,
    vm_allocation: Dict[str, int],
    prev_vm_allocation: Dict[str, int],
    vm_catalog: Dict[str, Dict],
    config: RewardConfig,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the total reward for a given state-action pair.
    
    Args:
        cpu_required: Required CPU cores (from demand)
        mem_required: Required memory in GB (from demand)
        cpu_threshold: Host CPU threshold (cores)
        mem_threshold: Host memory threshold (GB)
        cpu_allocated: Allocated CPU cores from VMs
        mem_allocated: Allocated memory in GB from VMs
        vm_allocation: Current VM allocation {vm_type: count}
        prev_vm_allocation: Previous VM allocation {vm_type: count}
        vm_catalog: VM specifications
        config: Reward configuration (scenario-specific weights)
        
    Returns:
        Tuple of (total_reward, reward_breakdown_dict)
    """
    # Compute individual components
    sla_penalty, sla_violated = compute_sla_violation(
        cpu_required, mem_required, 
        cpu_threshold, mem_threshold,  # Host thresholds
        cpu_allocated, mem_allocated,  # VM resources
        config
    )
    
    overflow_penalty = compute_overflow_penalty(
        cpu_required, mem_required, cpu_threshold, mem_threshold, config
    )
    
    switching_cost = compute_switching_cost(
        prev_vm_allocation, vm_allocation, vm_catalog
    )
    
    vm_cost = compute_vm_cost(vm_allocation, vm_catalog)
    
    # Compute overflow (what VMs need to handle)
    cpu_overflow = max(0, cpu_required - cpu_threshold)
    mem_overflow = max(0, mem_required - mem_threshold)
    
    # Over-provision penalty when overflow=0 but VMs > 0
    total_vm_vcpus = 0.0
    total_vm_mem = 0.0
    for vm_type, count in vm_allocation.items():
        if vm_type in vm_catalog and count > 0:
            vm_spec = vm_catalog[vm_type]
            total_vm_vcpus += count * vm_spec.get("vcpus", 0)
            total_vm_mem += count * vm_spec.get("memory_gb", 0)
    overprov_penalty = 0.0
    if cpu_overflow <= 0 and mem_overflow <= 0 and (total_vm_vcpus > 0 or total_vm_mem > 0):
        overprov_penalty = (
            total_vm_vcpus * config.overprov_penalty_per_vcpu
            + total_vm_mem * config.overprov_penalty_per_gb
        )
    
    efficiency_bonus = compute_efficiency_bonus(
        cpu_overflow, mem_overflow, cpu_allocated, mem_allocated
    )
    
    # Apply weights and compute total reward
    total_reward = (
        - config.alpha * sla_penalty
        - config.beta * overflow_penalty
        - config.gamma * switching_cost
        - config.delta * vm_cost
        - overprov_penalty
        + config.epsilon * efficiency_bonus
    )
    
    # Breakdown for logging/debugging
    breakdown = {
        "sla_penalty": sla_penalty,
        "sla_violated": float(sla_violated),
        "overflow_penalty": overflow_penalty,
        "switching_cost": switching_cost,
        "vm_cost": vm_cost,
        "overprov_penalty": overprov_penalty,
        "efficiency_bonus": efficiency_bonus,
        "total_reward": total_reward,
    }
    
    return total_reward, breakdown

