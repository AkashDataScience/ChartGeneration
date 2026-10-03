from src.core.workflow import run_workflow

if __name__ == "__main__":
    generation_model = "gemini-3.5-flash"
    reflection_model = "gemini-3.5-flash"
    image_basename = "drink_sales"
    dataset_path = "data/coffee_sales.csv"

    print("Welcome to ChartAgent! Type 'exit' to quit.")
    
    # Store the previous code so the agent can iterate on it
    previous_code = ""

    while True:
        user_instructions = input("\nWhat would you like to plot or modify? > ")
        if user_instructions.strip().lower() in ['exit', 'quit']:
            break
            
        print("\nStarting workflow...")
        result = run_workflow(
            dataset_path=dataset_path,
            user_instructions=user_instructions,
            generation_model=generation_model,
            reflection_model=reflection_model,
            image_basename=image_basename,
            previous_code=previous_code
        )
        
        # Save the best code from this run to use as context for the next request
        previous_code = result.get("code_v2") or result.get("code_v1")
        print("\nWorkflow complete! You can view the image generated in your directory.")
