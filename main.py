from src.core.workflow import run_workflow

if __name__ == "__main__":
    user_instructions = "Create a plot comparing Q1 coffee sales in 2024 and 2025 using the data in coffee_sales.csv."
    generation_model = "gemini-3.6-flash"
    reflection_model = "gemini-3.6-flash"
    image_basename = "drink_sales"

    # Run the complete agentic workflow
    run_workflow(
        dataset_path="data/coffee_sales.csv",
        user_instructions=user_instructions,
        generation_model=generation_model,
        reflection_model=reflection_model,
        image_basename=image_basename
    )
