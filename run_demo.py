from agent.databricks_optimizer_agent import DatabricksOptimizerAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('demo')

def setup_directories():
    """Create necessary directories for the demo."""
    dirs = [
        "agent",
        "jobs/1001",
        "jobs/2001", 
        "jobs/3001",
        "data",
        "visualizations"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def run_demo():
    """Run the Databricks Optimizer Agent demo."""
    logger.info("Starting Databricks Optimizer Agent Demo")
    
    # Make sure directories exist
    setup_directories()
    
    # Initialize the agent
    agent = DatabricksOptimizerAgent("config.json")
    
    # Load sample job data
    agent.load_job_data("data/job_performance_data.csv")
    
    # Run the agent
    result = agent.start()
    
    # Print results
    logger.info(f"Demo completed with status: {result['status']}")
    
    if result['status'] == 'completed':
        logger.info(f"Generated {result['suggestions']} optimization suggestions")
        logger.info(f"Optimization report available at: {result['report_path']}")
        logger.info(f"Visualizations created: {', '.join(result['visualizations'])}")
    else:
        logger.error(f"Demo failed with error: {result.get('error', 'Unknown error')}")
    
    return result

if __name__ == "__main__":
    result = run_demo()
    
    print("\n" + "="*50)
    print("DATABRICKS OPTIMIZER AGENT DEMO")
    print("="*50)
    
    if result['status'] == 'completed':
        print("\n✅ Demo completed successfully!")
        print(f"\n📊 Generated {result['suggestions']} optimization suggestions")
        print(f"\n📄 View the report at: {result['report_path']}")
        print("\n📈 Visualizations created:")
        for vis in result['visualizations']:
            print(f"   - {vis}")
    else:
        print("\n❌ Demo failed!")
        print(f"\nError: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*50)
    print("Next steps:")
    print("1. Review the generated optimization report")
    print("2. Check the visualizations for performance insights")
    print("3. Look at any pull requests created with optimization suggestions")
    print("="*50 + "\n")
