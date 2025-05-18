import os
import json
import pandas as pd
import logging
from datetime import datetime
import requests
from github import Github
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('databricks-optimizer')

class DatabricksOptimizerAgent:
    """
    An agent that monitors Databricks jobs, analyzes their performance, 
    and suggests or implements optimizations.
    """
    
    def __init__(self, config_path="config.json"):
        """Initialize the agent with configuration from a JSON file."""
        logger.info("Initializing Databricks Optimizer Agent")
        self.config = self._load_config(config_path)
        self.databricks_api_token = self.config.get('databricks_api_token')
        self.databricks_workspace_url = self.config.get('databricks_workspace_url')
        self.github_token = self.config.get('github_token')
        self.repo_name = self.config.get('repo_name')
        self.performance_history = {}
        
    def _load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            return {}
            
    def load_job_data(self, data_path):
        """
        Load job execution data from CSV file.
        
        Expected columns:
        - job_id: unique identifier for the job
        - job_name: name of the job
        - configuration: JSON string of job configuration
        - runtime_minutes: how long the job ran
        - start_time: when the job started
        - end_time: when the job completed
        - output: log output from the job
        """
        logger.info(f"Loading job data from {data_path}")
        try:
            self.job_data = pd.read_csv(data_path)
            logger.info(f"Loaded data for {len(self.job_data)} job runs")
            return self.job_data
        except Exception as e:
            logger.error(f"Error loading job data: {e}")
            return None
    
    def analyze_jobs(self):
        """Analyze job performance and identify optimization opportunities."""
        if not hasattr(self, 'job_data'):
            logger.error("No job data loaded. Call load_job_data() first.")
            return None
            
        results = []
        
        for job_id in self.job_data['job_id'].unique():
            job_runs = self.job_data[self.job_data['job_id'] == job_id]
            
            if len(job_runs) < 2:
                logger.info(f"Job {job_id} has insufficient run history for analysis")
                continue
                
            job_name = job_runs['job_name'].iloc[0]
            avg_runtime = job_runs['runtime_minutes'].mean()
            max_runtime = job_runs['runtime_minutes'].max()
            min_runtime = job_runs['runtime_minutes'].min()
            
            # Example analysis: Look for long-running jobs
            if avg_runtime > 30:  # If average runtime > 30 minutes
                recommendation = "Consider increasing cluster size or optimizing code"
                severity = "HIGH"
            elif max_runtime > avg_runtime * 1.5:  # If max runtime is 50% higher than average
                recommendation = "Job has inconsistent performance, check for data skew"
                severity = "MEDIUM"
            else:
                recommendation = "Performance acceptable"
                severity = "LOW"
                
            # Parse configuration for additional insights
            try:
                latest_config = json.loads(job_runs['configuration'].iloc[-1])
                cluster_config = latest_config.get('cluster_config', {})
                driver_size = cluster_config.get('driver_node_type_id', 'unknown')
                worker_size = cluster_config.get('node_type_id', 'unknown')
                num_workers = cluster_config.get('num_workers', 0)
                
                # Add more specific recommendations based on configuration
                if avg_runtime > 30 and num_workers < 4:
                    recommendation += "; increase worker count"
                
                # Check if there's error output that might indicate issues
                error_patterns = ['OutOfMemoryError', 'SparkException', 'Error:', 'Exception:']
                latest_output = job_runs['output'].iloc[-1]
                
                errors_found = [pattern for pattern in error_patterns if pattern in str(latest_output)]
                if errors_found:
                    recommendation += f"; fix errors: {', '.join(errors_found)}"
                    severity = "HIGH"
                    
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Could not parse configuration for job {job_id}")
                driver_size = worker_size = "unknown"
                num_workers = 0
                
            # Store analysis results
            results.append({
                'job_id': job_id,
                'job_name': job_name,
                'avg_runtime': avg_runtime,
                'max_runtime': max_runtime,
                'min_runtime': min_runtime,
                'driver_size': driver_size,
                'worker_size': worker_size,
                'num_workers': num_workers,
                'recommendation': recommendation,
                'severity': severity
            })
            
            # Track performance history for this job
            if job_id not in self.performance_history:
                self.performance_history[job_id] = []
            
            self.performance_history[job_id].append({
                'timestamp': datetime.now().isoformat(),
                'avg_runtime': avg_runtime,
                'recommendation': recommendation
            })
            
        self.analysis_results = pd.DataFrame(results)
        return self.analysis_results
    
    def suggest_optimizations(self):
        """Generate optimization suggestions based on analysis."""
        if not hasattr(self, 'analysis_results'):
            logger.error("No analysis results available. Call analyze_jobs() first.")
            return None
            
        # Sort by severity and runtime
        prioritized = self.analysis_results.sort_values(
            by=['severity', 'avg_runtime'], 
            ascending=[False, False]
        )
        
        suggestions = []
        
        for _, job in prioritized.iterrows():
            job_id = job['job_id']
            job_name = job['job_name']
            
            # Basic suggestion
            suggestion = {
                'job_id': job_id,
                'job_name': job_name,
                'current_config': {
                    'driver_size': job['driver_size'],
                    'worker_size': job['worker_size'],
                    'num_workers': job['num_workers']
                },
                'recommendation': job['recommendation'],
                'severity': job['severity']
            }
            
            # Generate specific configuration suggestions
            if 'increase cluster size' in job['recommendation'].lower():
                suggestion['suggested_config'] = {
                    'driver_size': job['driver_size'],  # Keep driver the same
                    'worker_size': job['worker_size'],  # Keep instance type the same
                    'num_workers': max(2, int(job['num_workers'] * 1.5))  # Increase workers by 50%
                }
                suggestion['code_changes'] = [
                    "Consider using broadcast joins for small tables",
                    "Evaluate partition count and size",
                    "Review any UDFs for optimization opportunities"
                ]
            elif 'data skew' in job['recommendation'].lower():
                suggestion['suggested_config'] = {
                    'driver_size': job['driver_size'],
                    'worker_size': job['worker_size'],
                    'num_workers': job['num_workers']  # Keep the same
                }
                suggestion['code_changes'] = [
                    "Add salting for skewed join keys",
                    "Consider repartitioning before join operations",
                    "Review any aggregations that might cause skew"
                ]
            else:
                suggestion['suggested_config'] = suggestion['current_config'].copy()
                suggestion['code_changes'] = []
                
            suggestions.append(suggestion)
            
        return suggestions
    
    def generate_optimization_report(self, output_path="optimization_report.html"):
        """Generate an HTML report with job analysis and optimization suggestions."""
        if not hasattr(self, 'analysis_results'):
            logger.error("No analysis results available. Call analyze_jobs() first.")
            return None
            
        suggestions = self.suggest_optimizations()
        
        # Create a simple HTML report
        html = "<html><head><title>Databricks Job Optimization Report</title>"
        html += "<style>body{font-family:Arial,sans-serif;margin:20px;}"
        html += "table{border-collapse:collapse;width:100%;margin-bottom:20px;}"
        html += "th,td{border:1px solid #ddd;padding:8px;text-align:left;}"
        html += "th{background-color:#f2f2f2;}"
        html += ".high{color:red;font-weight:bold;} .medium{color:orange;} .low{color:green;}"
        html += "</style></head><body>"
        
        html += f"<h1>Databricks Job Optimization Report</h1>"
        html += f"<p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        
        # Summary section
        html += "<h2>Summary</h2>"
        html += f"<p>Analyzed {len(self.analysis_results)} jobs.</p>"
        html += f"<p>Found {sum(self.analysis_results['severity'] == 'HIGH')} high priority issues.</p>"
        html += f"<p>Found {sum(self.analysis_results['severity'] == 'MEDIUM')} medium priority issues.</p>"
        
        # Job Analysis Table
        html += "<h2>Job Analysis</h2>"
        html += "<table><tr><th>Job Name</th><th>Avg Runtime (min)</th><th>Workers</th>"
        html += "<th>Severity</th><th>Recommendation</th></tr>"
        
        for _, job in self.analysis_results.iterrows():
            severity_class = job['severity'].lower()
            html += f"<tr><td>{job['job_name']}</td><td>{job['avg_runtime']:.1f}</td>"
            html += f"<td>{job['num_workers']}</td>"
            html += f"<td class='{severity_class}'>{job['severity']}</td>"
            html += f"<td>{job['recommendation']}</td></tr>"
            
        html += "</table>"
        
        # Detailed Suggestions
        html += "<h2>Optimization Suggestions</h2>"
        
        for suggestion in suggestions:
            html += f"<h3>Job: {suggestion['job_name']}</h3>"
            html += f"<p class='{suggestion['severity'].lower()}'>"
            html += f"Priority: {suggestion['severity']}</p>"
            html += "<h4>Current Configuration:</h4>"
            html += "<ul>"
            html += f"<li>Driver Size: {suggestion['current_config']['driver_size']}</li>"
            html += f"<li>Worker Size: {suggestion['current_config']['worker_size']}</li>"
            html += f"<li>Number of Workers: {suggestion['current_config']['num_workers']}</li>"
            html += "</ul>"
            
            html += "<h4>Suggested Configuration:</h4>"
            html += "<ul>"
            html += f"<li>Driver Size: {suggestion['suggested_config']['driver_size']}</li>"
            html += f"<li>Worker Size: {suggestion['suggested_config']['worker_size']}</li>"
            html += f"<li>Number of Workers: {suggestion['suggested_config']['num_workers']}</li>"
            html += "</ul>"
            
            if suggestion['code_changes']:
                html += "<h4>Suggested Code Changes:</h4>"
                html += "<ul>"
                for change in suggestion['code_changes']:
                    html += f"<li>{change}</li>"
                html += "</ul>"
                
            html += "<hr>"
            
        html += "</body></html>"
        
        with open(output_path, 'w') as file:
            file.write(html)
            
        logger.info(f"Generated optimization report at {output_path}")
        return output_path
    
    def visualize_job_performance(self, output_dir="./visualizations"):
        """Generate visualizations of job performance metrics."""
        if not hasattr(self, 'analysis_results'):
            logger.error("No analysis results available. Call analyze_jobs() first.")
            return None
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Job runtime comparison
        plt.figure(figsize=(12, 6))
        job_runtimes = self.analysis_results.sort_values('avg_runtime', ascending=False)
        plt.bar(job_runtimes['job_name'], job_runtimes['avg_runtime'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Average Job Runtime by Job')
        plt.xlabel('Job Name')
        plt.ylabel('Average Runtime (minutes)')
        plt.tight_layout()
        runtime_path = os.path.join(output_dir, 'job_runtimes.png')
        plt.savefig(runtime_path)
        
        # Workers vs Runtime scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.analysis_results['num_workers'], 
                   self.analysis_results['avg_runtime'],
                   s=100, alpha=0.7)
        
        for i, job in self.analysis_results.iterrows():
            plt.annotate(job['job_name'], 
                        (job['num_workers'], job['avg_runtime']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Job Runtime vs. Number of Workers')
        plt.xlabel('Number of Workers')
        plt.ylabel('Average Runtime (minutes)')
        plt.grid(True, linestyle='--', alpha=0.7)
        workers_path = os.path.join(output_dir, 'workers_vs_runtime.png')
        plt.savefig(workers_path)
        
        # Generate cost analysis
        plt.figure(figsize=(10, 6))
        # Assume a simple cost model: $0.5 per worker per hour
        hourly_rate = 0.5
        
        self.analysis_results['cost_estimate'] = (
            self.analysis_results['num_workers'] * 
            self.analysis_results['avg_runtime'] / 60 * 
            hourly_rate
        )
        
        plt.bar(self.analysis_results['job_name'], self.analysis_results['cost_estimate'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Estimated Cost per Job Run')
        plt.xlabel('Job Name')
        plt.ylabel('Estimated Cost ($)')
        plt.tight_layout()
        cost_path = os.path.join(output_dir, 'job_costs.png')
        plt.savefig(cost_path)
        
        logger.info(f"Generated visualizations in {output_dir}")
        return [runtime_path, workers_path, cost_path]
    
    def create_optimization_pr(self, job_id):
        """Create a Pull Request with suggested optimizations for a specific job."""
        if not self.github_token or not self.repo_name:
            logger.error("GitHub token or repo name not configured")
            return None
            
        # Get the suggestions for this job
        if not hasattr(self, 'analysis_results'):
            logger.error("No analysis results available. Call analyze_jobs() first.")
            return None
            
        suggestions = self.suggest_optimizations()
        job_suggestion = next((s for s in suggestions if s['job_id'] == job_id), None)
        
        if not job_suggestion:
            logger.error(f"No suggestions found for job {job_id}")
            return None
            
        try:
            # Connect to GitHub
            g = Github(self.github_token)
            repo = g.get_repo(self.repo_name)
            
            # Create a new branch
            default_branch = repo.default_branch
            branch_name = f"optimize-job-{job_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Get the latest commit on the default branch
            ref = repo.get_git_ref(f"heads/{default_branch}")
            sha = ref.object.sha
            
            # Create new branch
            repo.create_git_ref(f"refs/heads/{branch_name}", sha)
            
            # Create or update configuration file
            config_path = f"jobs/{job_id}/job_config.json"
            
            try:
                # Try to get existing file
                file = repo.get_contents(config_path, ref=branch_name)
                current_config = json.loads(file.decoded_content.decode())
                
                # Update config with suggestions
                if 'cluster_config' in current_config:
                    current_config['cluster_config']['num_workers'] = job_suggestion['suggested_config']['num_workers']
                else:
                    current_config['cluster_config'] = {
                        'driver_node_type_id': job_suggestion['suggested_config']['driver_size'],
                        'node_type_id': job_suggestion['suggested_config']['worker_size'],
                        'num_workers': job_suggestion['suggested_config']['num_workers']
                    }
                
                updated_config = json.dumps(current_config, indent=2)
                
                # Commit the changes
                repo.update_file(
                    config_path,
                    f"Optimize configuration for job {job_id}",
                    updated_config,
                    file.sha,
                    branch=branch_name
                )
                
            except Exception as e:
                # File doesn't exist, or other error
                logger.info(f"Creating new configuration file for job {job_id}")
                
                # Create a basic config structure
                new_config = {
                    "job_id": job_id,
                    "job_name": job_suggestion['job_name'],
                    "cluster_config": {
                        "driver_node_type_id": job_suggestion['suggested_config']['driver_size'],
                        "node_type_id": job_suggestion['suggested_config']['worker_size'],
                        "num_workers": job_suggestion['suggested_config']['num_workers']
                    }
                }
                
                repo.create_file(
                    config_path,
                    f"Add configuration for job {job_id}",
                    json.dumps(new_config, indent=2),
                    branch=branch_name
                )
                
            # Create a README with code optimization suggestions if there are any
            if job_suggestion['code_changes']:
                readme_path = f"jobs/{job_id}/OPTIMIZATIONS.md"
                readme_content = f"# Optimization Suggestions for {job_suggestion['job_name']}\n\n"
                readme_content += "## Suggested Code Changes\n\n"
                
                for change in job_suggestion['code_changes']:
                    readme_content += f"- {change}\n"
                
                try:
                    file = repo.get_contents(readme_path, ref=branch_name)
                    repo.update_file(
                        readme_path,
                        f"Update optimization suggestions for job {job_id}",
                        readme_content,
                        file.sha,
                        branch=branch_name
                    )
                except:
                    repo.create_file(
                        readme_path,
                        f"Add optimization suggestions for job {job_id}",
                        readme_content,
                        branch=branch_name
                    )
                    
            # Create the pull request
            pr_title = f"Optimize job {job_suggestion['job_name']} (ID: {job_id})"
            pr_body = f"""
            # Job Optimization PR
            
            This PR contains suggested optimizations for job **{job_suggestion['job_name']}**.
            
            ## Current Configuration
            - Driver Size: {job_suggestion['current_config']['driver_size']}
            - Worker Size: {job_suggestion['current_config']['worker_size']}
            - Workers: {job_suggestion['current_config']['num_workers']}
            
            ## Suggested Configuration
            - Driver Size: {job_suggestion['suggested_config']['driver_size']}
            - Worker Size: {job_suggestion['suggested_config']['worker_size']}
            - Workers: {job_suggestion['suggested_config']['num_workers']}
            
            ## Recommendation
            {job_suggestion['recommendation']}
            
            ## Performance Analysis
            - Average Runtime: {self.analysis_results[self.analysis_results['job_id'] == job_id]['avg_runtime'].values[0]:.1f} minutes
            - Priority: {job_suggestion['severity']}
            
            Generated automatically by the Databricks Optimizer Agent.
            """
            
            pr = repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=branch_name,
                base=default_branch
            )
            
            logger.info(f"Created PR #{pr.number}: {pr.html_url}")
            return pr.html_url
            
        except Exception as e:
            logger.error(f"Error creating PR: {e}")
            return None
    
    def start(self):
        """Start the agent's monitoring and optimization loop."""
        logger.info("Starting Databricks Optimizer Agent")
        
        # For demo purposes, we'll just run analysis once
        if hasattr(self, 'job_data'):
            self.analyze_jobs()
            suggestions = self.suggest_optimizations()
            
            logger.info(f"Generated {len(suggestions)} optimization suggestions")
            
            # Generate the report
            report_path = self.generate_optimization_report()
            logger.info(f"Optimization report generated at {report_path}")
            
            # Create visualizations
            vis_paths = self.visualize_job_performance()
            logger.info(f"Generated {len(vis_paths)} visualization images")
            
            # For demo, create a PR for the job with highest severity
            if suggestions:
                highest_priority = sorted(suggestions, key=lambda s: 
                    (0 if s['severity'] == 'HIGH' else 
                     1 if s['severity'] == 'MEDIUM' else 2)
                )[0]
                
                logger.info(f"Creating PR for highest priority job: {highest_priority['job_name']}")
                pr_url = self.create_optimization_pr(highest_priority['job_id'])
                
                if pr_url:
                    logger.info(f"Created PR: {pr_url}")
                    
            return {
                'status': 'completed',
                'suggestions': len(suggestions),
                'report_path': report_path,
                'visualizations': vis_paths
            }
        else:
            logger.error("No job data loaded. Call load_job_data() first.")
            return {'status': 'failed', 'error': 'No job data loaded'}

if __name__ == "__main__":
    agent = DatabricksOptimizerAgent("config.json")
    agent.load_job_data("job_performance_data.csv")
    result = agent.start()
    print(f"Agent completed with status: {result['status']}")
