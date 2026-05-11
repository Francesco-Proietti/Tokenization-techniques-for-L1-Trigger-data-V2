import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Ask for the CSV file path
    csv_path = input("Enter the path of the CSV file: ").strip()

    # Verify that the file exists
    if not os.path.isfile(csv_path):
        print("Error: file not found.")
        return

    # Ask for the output folder
    output_dir = "loss_plots" 

    try:
        # Read the CSV
        df = pd.read_csv(csv_path)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(df["step"], df["train_loss"], marker='o')

        plt.xlabel("step")
        plt.ylabel("train_loss")
        plt.title(f"train_loss vs step")
        plt.grid(True)

        # Save figure
        output_path = os.path.join(output_dir, "plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        print(f"\nPlot saved in: {output_path}")

        # Show the plot
        plt.show()

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()