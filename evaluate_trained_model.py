#!/usr/bin/env python3
"""
Model Evaluation Script for Enhanced Duckietown RL

This script loads and evaluates the trained model to demonstrate its performance.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import the environment from our training script
from train_simple_reliable import SimpleDuckietownEnv, SimplePolicy


class ModelEvaluator:
    """Evaluates trained models."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_data = self._load_model()
        self.env = SimpleDuckietownEnv()
        self.policy = self._create_policy_from_model()
        
    def _load_model(self):
        """Load the trained model."""
        try:
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)
            print(f"✅ Loaded model from {self.model_path}")
            return model_data
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    
    def _create_policy_from_model(self):
        """Create policy from loaded model data."""
        if not self.model_data:
            return SimplePolicy(self.env.action_space)
        
        policy = SimplePolicy(self.env.action_space)
        
        # Load trained parameters
        params = self.model_data.get('policy_params', {})
        policy.steering_bias = params.get('steering_bias', 0.0)
        policy.throttle_bias = params.get('throttle_bias', 0.5)
        policy.epsilon = params.get('epsilon', 0.05)  # Low epsilon for evaluation
        policy.learning_rate = params.get('learning_rate', 0.01)
        
        print(f"📊 Loaded policy parameters:")
        print(f"  Steering bias: {policy.steering_bias:.4f}")
        print(f"  Throttle bias: {policy.throttle_bias:.4f}")
        print(f"  Exploration rate: {policy.epsilon:.4f}")
        
        return policy
    
    def evaluate(self, num_episodes: int = 20, render: bool = False):
        """Evaluate the trained model."""
        print(f"🧪 Evaluating model for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        episode_details = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Track episode statistics
            max_position = 0
            min_lane_deviation = float('inf')
            max_lane_deviation = 0
            avg_speed = 0
            
            while not done and episode_length < 500:
                # Get action from trained policy
                action = self.policy.get_action(obs)
                
                # Take step
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Track statistics
                max_position = max(max_position, info.get('position', 0))
                lane_dev = abs(info.get('lane_deviation', 0))
                min_lane_deviation = min(min_lane_deviation, lane_dev)
                max_lane_deviation = max(max_lane_deviation, lane_dev)
                avg_speed += info.get('speed', 0)
                
                if render:
                    # Simple text-based rendering
                    if episode_length % 10 == 0:  # Print every 10 steps
                        print(f"  Step {episode_length}: Reward={reward:.3f}, "
                              f"Position={info.get('position', 0):.2f}, "
                              f"Lane Dev={info.get('lane_deviation', 0):.3f}")
            
            avg_speed = avg_speed / episode_length if episode_length > 0 else 0
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            episode_details.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'length': episode_length,
                'max_position': max_position,
                'min_lane_deviation': min_lane_deviation,
                'max_lane_deviation': max_lane_deviation,
                'avg_speed': avg_speed,
                'success': episode_length >= 100 and episode_reward > 50  # Define success criteria
            })
            
            print(f"Episode {episode + 1:2d}: Reward={episode_reward:7.2f}, "
                  f"Length={episode_length:3d}, Position={max_position:6.2f}, "
                  f"Success={'✅' if episode_details[-1]['success'] else '❌'}")
        
        # Calculate statistics
        stats = self._calculate_statistics(episode_rewards, episode_lengths, episode_details)
        
        # Print results
        self._print_evaluation_results(stats)
        
        # Save evaluation results
        self._save_evaluation_results(stats, episode_details)
        
        # Create evaluation plots
        self._create_evaluation_plots(episode_rewards, episode_lengths, episode_details)
        
        return stats
    
    def _calculate_statistics(self, rewards, lengths, details):
        """Calculate evaluation statistics."""
        stats = {
            'num_episodes': len(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'median_reward': np.median(rewards),
            
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            
            'success_rate': np.mean([d['success'] for d in details]),
            'avg_max_position': np.mean([d['max_position'] for d in details]),
            'avg_min_lane_deviation': np.mean([d['min_lane_deviation'] for d in details]),
            'avg_max_lane_deviation': np.mean([d['max_lane_deviation'] for d in details]),
            'avg_speed': np.mean([d['avg_speed'] for d in details]),
            
            'model_path': self.model_path,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return stats
    
    def _print_evaluation_results(self, stats):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("🎯 MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"📊 Episodes Evaluated: {stats['num_episodes']}")
        print(f"🏆 Success Rate: {stats['success_rate']*100:.1f}%")
        print()
        
        print("📈 REWARD STATISTICS:")
        print(f"  Mean Reward: {stats['mean_reward']:8.2f} ± {stats['std_reward']:.2f}")
        print(f"  Best Reward: {stats['max_reward']:8.2f}")
        print(f"  Worst Reward: {stats['min_reward']:7.2f}")
        print(f"  Median Reward: {stats['median_reward']:6.2f}")
        print()
        
        print("📏 EPISODE LENGTH STATISTICS:")
        print(f"  Mean Length: {stats['mean_length']:8.1f} ± {stats['std_length']:.1f} steps")
        print(f"  Longest Episode: {stats['max_length']:5.0f} steps")
        print(f"  Shortest Episode: {stats['min_length']:4.0f} steps")
        print()
        
        print("🚗 DRIVING PERFORMANCE:")
        print(f"  Average Max Position: {stats['avg_max_position']:6.2f}")
        print(f"  Average Speed: {stats['avg_speed']:13.3f}")
        print(f"  Best Lane Keeping: {stats['avg_min_lane_deviation']:7.3f}")
        print(f"  Worst Lane Deviation: {stats['avg_max_lane_deviation']:5.3f}")
        print()
        
        # Performance assessment
        if stats['success_rate'] > 0.8:
            performance = "🌟 EXCELLENT"
        elif stats['success_rate'] > 0.6:
            performance = "✅ GOOD"
        elif stats['success_rate'] > 0.4:
            performance = "⚠️ FAIR"
        else:
            performance = "❌ NEEDS IMPROVEMENT"
        
        print(f"🎯 Overall Performance: {performance}")
        print("="*60)
    
    def _save_evaluation_results(self, stats, details):
        """Save evaluation results to file."""
        # Create evaluation directory
        eval_dir = Path("evaluation")
        eval_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results = {
            'statistics': stats,
            'episode_details': details,
            'model_info': self.model_data.get('training_results', {}) if self.model_data else {}
        }
        
        results_path = eval_dir / f"evaluation_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Evaluation results saved to {results_path}")
    
    def _create_evaluation_plots(self, rewards, lengths, details):
        """Create evaluation plots."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            episodes = range(1, len(rewards) + 1)
            
            # Episode rewards
            ax1.plot(episodes, rewards, 'bo-', alpha=0.7, markersize=4)
            ax1.axhline(np.mean(rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(rewards):.2f}')
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Episode lengths
            ax2.plot(episodes, lengths, 'go-', alpha=0.7, markersize=4)
            ax2.axhline(np.mean(lengths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(lengths):.1f}')
            ax2.set_title('Episode Lengths')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Success rate over time (rolling window)
            window_size = min(5, len(details))
            success_rolling = []
            for i in range(len(details)):
                start_idx = max(0, i - window_size + 1)
                window_success = [d['success'] for d in details[start_idx:i+1]]
                success_rolling.append(np.mean(window_success))
            
            ax3.plot(episodes, success_rolling, 'mo-', alpha=0.7, markersize=4)
            ax3.set_title(f'Success Rate (Rolling Window = {window_size})')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Success Rate')
            ax3.set_ylim(0, 1.1)
            ax3.grid(True, alpha=0.3)
            
            # Performance scatter plot
            max_positions = [d['max_position'] for d in details]
            colors = ['green' if d['success'] else 'red' for d in details]
            ax4.scatter(max_positions, rewards, c=colors, alpha=0.7)
            ax4.set_title('Reward vs Max Position')
            ax4.set_xlabel('Max Position Reached')
            ax4.set_ylabel('Episode Reward')
            ax4.grid(True, alpha=0.3)
            
            # Add legend for scatter plot
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', label='Success'),
                             Patch(facecolor='red', label='Failure')]
            ax4.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            # Save plot
            eval_dir = Path("evaluation")
            eval_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = eval_dir / f"evaluation_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Evaluation plots saved to {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Could not create evaluation plots: {e}")
    
    def compare_with_training(self):
        """Compare evaluation results with training performance."""
        if not self.model_data or 'training_results' not in self.model_data:
            print("⚠️ No training data available for comparison")
            return
        
        training_results = self.model_data['training_results']
        
        print("\n" + "="*60)
        print("📊 TRAINING vs EVALUATION COMPARISON")
        print("="*60)
        
        print(f"Training Episodes: {training_results.get('total_episodes', 'N/A')}")
        print(f"Training Time: {training_results.get('training_time_seconds', 0)/60:.1f} minutes")
        print(f"Best Training Reward: {training_results.get('best_reward', 'N/A'):.2f}")
        print(f"Final Training Avg: {training_results.get('final_avg_reward', 'N/A'):.2f}")
        print()
        
        # Calculate recent training performance
        all_rewards = training_results.get('all_rewards', [])
        if len(all_rewards) >= 50:
            recent_training_avg = np.mean(all_rewards[-50:])
            print(f"Recent Training Avg (last 50): {recent_training_avg:.2f}")
        
        print("="*60)


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Trained Duckietown RL Model")
    parser.add_argument("--model", help="Path to trained model file")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes (text output)")
    
    args = parser.parse_args()
    
    # Find the most recent model if none specified
    if not args.model:
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.json"))
            if model_files:
                # Sort by modification time and get the most recent
                args.model = str(sorted(model_files, key=lambda x: x.stat().st_mtime)[-1])
                print(f"🔍 Using most recent model: {args.model}")
            else:
                print("❌ No model files found in models/ directory")
                return
        else:
            print("❌ Models directory not found")
            return
    
    print("🧪 Enhanced Duckietown RL - Model Evaluation")
    print("=" * 55)
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model)
    
    # Run evaluation
    stats = evaluator.evaluate(num_episodes=args.episodes, render=args.render)
    
    # Compare with training
    evaluator.compare_with_training()
    
    print("\n✅ Evaluation completed!")
    print("📊 Check evaluation/ for detailed results and plots")


if __name__ == "__main__":
    main()