import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add the src directory to the system path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

def display_geoimagenet_results():
    # Example data (replace with actual evaluation data)
    data = {
        'Model': ['Sup. Learning (Scratch)', 'Geoloc. Learning', 'MoCo-V2', 'MoCo-V2+Geo'],
        'Backbone': ['ResNet50', 'ResNet50', 'ResNet50', 'ResNet50'],
        'Top-1 Accuracy': [2.04, 2.26, 18.51, 29.96],
        'Top-5 Accuracy': [4.11, 19.33, 27.67, 38.71]
    }

    df = pd.DataFrame(data)

    # Display the results using tabulate for a nicer format
    print(tabulate(df, headers='keys', tablefmt='psql'))

    # Create a table for display using matplotlib
    fig, ax = plt.subplots(figsize=(10, 2))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Set table font size
    table.set_fontsize(14)
    table.scale(1.5, 1.5)  # Adjust the scale as needed

    # Save the table as an image
    plt.savefig('/Users/nithinrajulapati/Downloads/PROJECT 1/z_FINAL_RESULTS/geoimagenet_results_table.png')

    # Show the table
    plt.show()

if __name__ == "__main__":
    display_geoimagenet_results()
