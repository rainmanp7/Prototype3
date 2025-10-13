# github_safe_testbed.py
"""
🧪 HOLOLIFEX6 PROTOTYPE3 - GITHUB-SAFE TESTING HARNESS
Safe, incremental testing with self-contained implementation
Runs within GitHub Actions limits (7GB RAM, 6 hours)
"""

import time
import numpy as np
import psutil
import os
from typing import Dict, List, Any, Optional
import json


class PulseCoupledEntity:
    """Pulse-coupled entity with phase dynamics"""
    
    def __init__(self, entity_id: str, domain: str, base_frequency: float = 0.02):
        self.entity_id = entity_id
        self.domain = domain
        self.base_frequency = base_frequency
        self.phase = np.random.random()
        self.state_vector = np.random.randn(8) * 0.1
        self.coupling_strength = 0.1
        
    def evolve_phase(self):
        """Evolve phase forward"""
        self.phase = (self.phase + self.base_frequency) % 1.0
        
    def couple_to(self, other_phase: float, strength: float = 0.05):
        """Couple to another entity's phase"""
        phase_diff = other_phase - self.phase
        self.phase += strength * np.sin(2 * np.pi * phase_diff)
        self.phase = self.phase % 1.0
        
    def generate_insight(self) -> Dict[str, Any]:
        """Generate insight when phase threshold crossed"""
        if self.phase > 0.8:
            return {
                'entity': self.entity_id,
                'domain': self.domain,
                'action': 'insight_generated',
                'confidence': self.phase,
                'phase': self.phase
            }
        return {}


class Lightweight4DSelector:
    """Lightweight 4D decision selector"""
    
    def __init__(self, num_entities: int, dim: int = 8):
        self.num_entities = num_entities
        self.dim = dim
        self.weights = np.random.randn(dim, 4) * 0.1
        
    def select_actions(self, state_matrix: np.ndarray) -> np.ndarray:
        """Select actions based on 4D projection"""
        batch_size = state_matrix.shape[0]
        reshaped = state_matrix.reshape(batch_size, -1)
        
        # Simple projection to 4D space
        if reshaped.shape[1] >= self.dim:
            projected = reshaped[:, :self.dim] @ self.weights
        else:
            padded = np.zeros((batch_size, self.dim))
            padded[:, :reshaped.shape[1]] = reshaped
            projected = padded @ self.weights
            
        # Softmax over actions
        exp_proj = np.exp(projected - np.max(projected, axis=1, keepdims=True))
        return exp_proj / np.sum(exp_proj, axis=1, keepdims=True)


class ScalableEntityNetwork:
    """Scalable network of pulse-coupled entities"""
    
    def __init__(self, decision_model: Lightweight4DSelector):
        self.entities: List[PulseCoupledEntity] = []
        self.decision_model = decision_model
        self.coherence_history = []
        
    def add_entity(self, entity: PulseCoupledEntity):
        """Add entity to network"""
        self.entities.append(entity)
        
    def get_state_matrix(self) -> np.ndarray:
        """Get current state matrix"""
        states = [e.state_vector for e in self.entities]
        return np.array(states).reshape(1, len(self.entities), -1)
        
    def evolve_step(self, system_state: Dict[str, float]) -> List[Dict[str, Any]]:
        """Single evolution step"""
        insights = []
        
        # Evolve all entity phases
        for entity in self.entities:
            entity.evolve_phase()
            
        # Couple entities
        avg_phase = np.mean([e.phase for e in self.entities])
        for entity in self.entities:
            entity.couple_to(avg_phase, strength=0.05)
            
        # Generate insights
        for entity in self.entities:
            insight = entity.generate_insight()
            if insight:
                insights.append(insight)
                
        # Update coherence
        phases = np.array([e.phase for e in self.entities])
        coherence = 1.0 - np.std(phases)
        self.coherence_history.append(coherence)
        
        return insights
        
    def get_coherence(self) -> float:
        """Get current coherence"""
        if not self.coherence_history:
            return 0.0
        return self.coherence_history[-1]
        
    def measure_performance(self) -> Dict[str, float]:
        """Measure performance metrics"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        start = time.time()
        self.get_state_matrix()
        step_time_ms = (time.time() - start) * 1000
        
        return {
            'memory_mb': memory_mb,
            'step_time_ms': step_time_ms,
            'entity_count': len(self.entities),
            'coherence': self.get_coherence()
        }


class Prototype2:
    """Golden master prototype for baseline testing"""
    
    def __init__(self):
        self.network: Optional[ScalableEntityNetwork] = None
        self.system_state = {'memory_usage': 0.7, 'cpu_load': 0.6, 'coherence': 0.0}
        
    def initialize(self):
        """Initialize with 16 entities"""
        domains = ['physical', 'temporal', 'semantic', 'network']
        entities = []
        
        for i in range(16):
            domain = domains[i % len(domains)]
            freq = 0.015 + (i * 0.002)
            entity_id = f"{domain[:3].upper()}-{i+1:02d}"
            entities.append(PulseCoupledEntity(entity_id, domain, freq))
            
        decision_model = Lightweight4DSelector(num_entities=16, dim=8)
        self.network = ScalableEntityNetwork(decision_model)
        
        for entity in entities:
            self.network.add_entity(entity)


class SafeTester:
    """Incremental testing that won't break GitHub"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def log(self, message: str):
        """Safe logging with timing"""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:6.1f}s] {message}")
    
    def memory_check(self) -> bool:
        """Check if we're approaching GitHub memory limits"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > 5000:
            self.log(f"⚠️  MEMORY WARNING: {memory_mb:.1f}MB - approaching GitHub limits")
            return False
        return True
    
    def run_baseline_test(self) -> Dict[str, Any]:
        """Test 1: Baseline with proven 16-entity system"""
        self.log("🧪 TEST 1: Baseline 16-entity validation")
        
        prototype = Prototype2()
        prototype.initialize()
        
        baseline_metrics = []
        for cycle in range(100):
            insights = prototype.network.evolve_step(prototype.system_state)
            
            if cycle % 10 == 0:
                metrics = prototype.network.measure_performance()
                metrics['cycle'] = cycle
                metrics['insights'] = len(insights)
                baseline_metrics.append(metrics)
                
                if not self.memory_check():
                    self.log("🛑 Stopping test - memory limits approached")
                    break
        
        result = {
            'test_name': 'baseline_16_entities',
            'entity_count': 16,
            'cycles_completed': len(baseline_metrics) * 10,
            'final_coherence': prototype.network.get_coherence(),
            'avg_memory_mb': np.mean([m['memory_mb'] for m in baseline_metrics]),
            'avg_step_time_ms': np.mean([m['step_time_ms'] for m in baseline_metrics]),
            'total_insights': sum(m['insights'] for m in baseline_metrics),
            'status': 'completed_safe'
        }
        
        self.results.append(result)
        self.log(f"✅ Baseline completed: {result['avg_memory_mb']:.1f}MB memory")
        return result
    
    def run_small_scale_test(self, entity_count: int = 32) -> Dict[str, Any]:
        """Test 2: Small scale increase"""
        self.log(f"🧪 TEST 2: Small scale - {entity_count} entities")
        
        domains = ['physical', 'temporal', 'semantic', 'network', 
                  'spatial', 'emotional', 'social', 'creative']
        
        entities = []
        for i in range(entity_count):
            domain = domains[i % len(domains)]
            freq = 0.015 + (i * 0.001)
            entity_id = f"{domain[:3].upper()}-{i+1:02d}"
            entities.append(PulseCoupledEntity(entity_id, domain, freq))
        
        decision_model = Lightweight4DSelector(num_entities=len(entities), dim=8)
        network = ScalableEntityNetwork(decision_model)
        
        for entity in entities:
            network.add_entity(entity)
        
        system_state = {'memory_usage': 0.7, 'cpu_load': 0.6, 'coherence': 0.0}
        scale_metrics = []
        
        for cycle in range(50):
            insights = network.evolve_step(system_state)
            
            if cycle % 10 == 0:
                metrics = network.measure_performance()
                metrics['cycle'] = cycle
                metrics['insights'] = len(insights)
                scale_metrics.append(metrics)
                
                if not self.memory_check():
                    self.log("🛑 Stopping scale test - memory limits")
                    break
        
        result = {
            'test_name': f'scale_{entity_count}_entities',
            'entity_count': entity_count,
            'cycles_completed': len(scale_metrics) * 10,
            'final_coherence': network.get_coherence(),
            'avg_memory_mb': np.mean([m['memory_mb'] for m in scale_metrics]),
            'avg_step_time_ms': np.mean([m['step_time_ms'] for m in scale_metrics]),
            'total_insights': sum(m['insights'] for m in scale_metrics),
            'scaling_ratio': entity_count / 16,
            'status': 'completed_safe'
        }
        
        self.results.append(result)
        self.log(f"✅ Scale test completed: {result['avg_memory_mb']:.1f}MB memory")
        return result

    def run_scaling_sweep(self) -> List[Dict[str, Any]]:
        """Test 3: Progressive scaling sweep"""
        self.log("🧪 TEST 3: Progressive scaling sweep")
        
        entity_counts = [16, 32, 64, 128]
        sweep_results = []
        
        for entity_count in entity_counts:
            self.log(f"   Testing {entity_count} entities...")
            
            if entity_count == 16:
                result = self.results[0] if self.results else self.run_baseline_test()
            else:
                result = self.run_small_scale_test(entity_count)
            
            sweep_results.append(result)
            
            if result['status'] != 'completed_safe':
                break
        
        # Calculate scaling efficiency
        baseline_memory = sweep_results[0]['avg_memory_mb']
        for result in sweep_results:
            if result['entity_count'] > 16:
                expected_linear = baseline_memory * (result['entity_count'] / 16)
                actual_memory = result['avg_memory_mb']
                efficiency = (expected_linear - actual_memory) / expected_linear * 100
                result['scaling_efficiency'] = efficiency
                result['scaling_class'] = self.classify_scaling(efficiency)
        
        return sweep_results

    def classify_scaling(self, efficiency: float) -> str:
        """Classify scaling performance"""
        if efficiency > 20:
            return "BETTER_THAN_LINEAR"
        elif efficiency > 0:
            return "LINEAR" 
        else:
            return "SUB_LINEAR"

    def save_results(self):
        """Save all results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"scaling_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"💾 Results saved to: {filename}")
        return filename


def main():
    """Main safe testing sequence"""
    print("🚀 HOLOLIFEX6 PROTOTYPE3 - GITHUB-SAFE TESTING HARNESS")
    print("=" * 60)
    
    tester = SafeTester()
    
    try:
        # Phase 1: Baseline validation
        baseline_result = tester.run_baseline_test()
        
        # Phase 2: Small scale test
        if baseline_result['status'] == 'completed_safe':
            scale_result = tester.run_small_scale_test(entity_count=32)
            
            # Phase 3: Progressive scaling sweep
            if scale_result['status'] == 'completed_safe':
                sweep_results = tester.run_scaling_sweep()
        
        # Save all results
        results_file = tester.save_results()
        
        # Print summary
        print("\n📊 PROTOTYPE3 TESTING SUMMARY:")
        print("=" * 40)
        for result in tester.results:
            print(f"🧪 {result['test_name']}:")
            print(f"   Entities: {result['entity_count']}")
            print(f"   Memory: {result['avg_memory_mb']:.1f}MB")
            print(f"   Step Time: {result['avg_step_time_ms']:.1f}ms")
            print(f"   Coherence: {result['final_coherence']:.3f}")
            if 'scaling_efficiency' in result:
                print(f"   Scaling: {result['scaling_class']} ({result['scaling_efficiency']:+.1f}%)")
            print(f"   Status: {result['status']}")
            print()
        
        print(f"✅ All tests completed safely!")
        print(f"📁 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        tester.save_results()
        raise


if __name__ == "__main__":
    main()
