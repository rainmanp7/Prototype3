# github_safe_testbed.py
"""
🧪 HOLOLIFEX6 PROTOTYPE3 - GITHUB-SAFE TESTING HARNESS
Safe, incremental testing that imports our golden master
Runs within GitHub Actions limits (7GB RAM, 6 hours)
"""

import time
import numpy as np
import psutil
import os
from typing import Dict, List, Any
import json

# Import our proven golden master
from prototype2_scaling import Prototype2, PulseCoupledEntity, ScalableEntityNetwork, Lightweight4DSelector

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
        
        # GitHub limit: 7GB (7168MB) - leave 2GB buffer for safety
        if memory_mb > 5000:
            self.log(f"⚠️  MEMORY WARNING: {memory_mb:.1f}MB - approaching GitHub limits")
            return False
        return True
    
    def run_baseline_test(self) -> Dict[str, Any]:
        """Test 1: Baseline with our proven 16-entity system"""
        self.log("🧪 TEST 1: Baseline 16-entity validation")
        
        # Use the exact same setup as golden master
        prototype = Prototype2()
        prototype.initialize()
        
        # Run minimal cycles to validate everything works
        baseline_metrics = []
        for cycle in range(100):  # Very safe - quick validation
            insights = prototype.network.evolve_step(prototype.system_state)
            
            # Track metrics every 10 cycles
            if cycle % 10 == 0:
                metrics = prototype.network.measure_performance()
                metrics['cycle'] = cycle
                metrics['insights'] = len(insights)
                baseline_metrics.append(metrics)
                
                # Memory safety check
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
        """Test 2: Small scale increase (2x entities)"""
        self.log(f"🧪 TEST 2: Small scale - {entity_count} entities")
        
        # Build custom network with more entities
        domains = ['physical', 'temporal', 'semantic', 'network', 
                  'spatial', 'emotional', 'social', 'creative']
        
        # Create entities (reusing golden master entity class)
        entities = []
        for domain_idx, domain in enumerate(domains):
            # Distribute entities across domains
            entities_per_domain = max(1, entity_count // len(domains))
            for entity_idx in range(entities_per_domain):
                freq = 0.015 + (domain_idx * 0.002) + (entity_idx * 0.001)
                entity_id = f"{domain[:3].upper()}-{entity_idx+1:02d}"
                entities.append(PulseCoupledEntity(entity_id, domain, freq))
        
        # Use same decision model as golden master
        decision_model = Lightweight4DSelector(num_entities=len(entities), dim=8)
        network = ScalableEntityNetwork(decision_model)
        
        # Add all entities
        for entity in entities:
            network.add_entity(entity)
        
        # Run safe test
        system_state = {'memory_usage': 0.7, 'cpu_load': 0.6, 'coherence': 0.0}
        scale_metrics = []
        
        for cycle in range(50):  # Very conservative
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
        """Test 3: Progressive scaling sweep (16 → 128 entities)"""
        self.log("🧪 TEST 3: Progressive scaling sweep")
        
        entity_counts = [16, 32, 64, 128]  # Safe progression
        sweep_results = []
        
        for entity_count in entity_counts:
            self.log(f"   Testing {entity_count} entities...")
            
            # Reuse the small scale test logic
            if entity_count == 16:
                result = self.run_baseline_test()
            else:
                result = self.run_small_scale_test(entity_count)
            
            sweep_results.append(result)
            
            # Stop if we hit memory limits
            if result['status'] != 'completed_safe':
                break
        
        # Calculate scaling efficiency
        for result in sweep_results:
            if result['entity_count'] > 16:
                baseline_memory = sweep_results[0]['avg_memory_mb']
                expected_linear = baseline_memory * (result['entity_count'] / 16)
                actual_memory = result['avg_memory_mb']
                efficiency = (expected_linear - actual_memory) / expected_linear * 100
                result['scaling_efficiency'] = efficiency
                result['scaling_class'] = self.classify_scaling(efficiency)
        
        self.results.extend(sweep_results)
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
        
        # Phase 2: Small scale test (only if baseline successful)
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
        # Save partial results
        tester.save_results()
        raise

if __name__ == "__main__":
    main()