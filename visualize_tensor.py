import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


def load_tensor(file_path):
    """Load a numpy tensor from a .npy file."""
    return np.load(file_path)


def plot_tensor_interactive(tensor, title=None, vmax=None, vmin=None, figsize=(12, 8), cmap='viridis', save_path=None):
    """Plot a 2D tensor as an interactive heatmap with hover values."""
    # Create hover text with row, column, and value information
    hover_text = []
    for i in range(tensor.shape[0]):
        row_text = []
        for j in range(tensor.shape[1]):
            value = tensor[i, j]
            row_text.append(f'Row: {i}<br>Column: {j}<br>Value: {value:.6f}')
        hover_text.append(row_text)
    
    fig = go.Figure(data=go.Heatmap(
        z=tensor,
        colorscale=cmap,
        zmax=vmax,
        zmin=vmin,
        hoverongaps=False,
        hovertemplate='%{text}<extra></extra>',
        text=hover_text
    ))
    
    fig.update_layout(
        title=title if title else "Tensor Visualization",
        xaxis_title="Column",
        yaxis_title="Row",
        width=figsize[0] * 100,
        height=figsize[1] * 100
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive plot to {save_path}")
    else:
        fig.show()
    
    return fig


def plot_tensor(tensor, title=None, vmax=None, vmin=None, figsize=(12, 8), cmap='viridis', show_colorbar=True, save_path=None):
    """Plot a 2D tensor as a heatmap."""
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        tensor,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        cbar=show_colorbar,
        square=False,
        linewidths=0.1,
        linecolor='gray',
    )
    if title:
        plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize a .npy tensor as a matrix heatmap.")
    parser.add_argument('file', type=str, help='Path to the .npy tensor file')
    parser.add_argument('--title', type=str, default=None, help='Title for the plot')
    parser.add_argument('--vmax', type=float, default=None, help='Max value for color scale')
    parser.add_argument('--vmin', type=float, default=None, help='Min value for color scale')
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 8], help='Figure size (width height)')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap for the heatmap')
    parser.add_argument('--no_colorbar', action='store_true', help='Hide colorbar')
    parser.add_argument('--save', type=str, default=None, help='Path to save the plot (if not shown interactively)')
    parser.add_argument('--static', action='store_true', help='Use static matplotlib heatmap (default is interactive)')
    args = parser.parse_args()

    try:
        tensor = load_tensor(args.file)
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found!")
        print("Usage: python visualize_tensor.py <tensor_file.npy> [options]")
        return
    except Exception as e:
        print(f"Error loading tensor from '{args.file}': {e}")
        return
    print(f"Loaded tensor from {args.file} with shape {tensor.shape} and dtype {tensor.dtype}")
    print(f"Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")

    if args.static:
        # Use static matplotlib version
        plot_tensor(
            tensor,
            title=args.title,
            vmax=args.vmax,
            vmin=args.vmin,
            figsize=tuple(args.figsize),
            cmap=args.cmap,
            show_colorbar=not args.no_colorbar,
            save_path=args.save,
        )
    else:
        # Use interactive plotly version (default)
        save_path = args.save.replace('.png', '.html') if args.save and args.save.endswith('.png') else args.save
        plot_tensor_interactive(
            tensor,
            title=args.title,
            vmax=args.vmax,
            vmin=args.vmin,
            figsize=tuple(args.figsize),
            cmap=args.cmap,
            save_path=save_path,
        )


if __name__ == "__main__":
    main() 