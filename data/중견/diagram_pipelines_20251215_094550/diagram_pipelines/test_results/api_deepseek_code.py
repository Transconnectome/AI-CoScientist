
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_transformer():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    encoder_color = '#1f77b4'
    decoder_color = '#ff7f0e'
    attention_color = '#2ca02c'
    norm_color = '#9467bd'
    ff_color = '#8c564b'
    embed_color = '#17becf'
    final_color = '#e377c2'
    
    # Encoder components
    encoder_x = 2.0
    encoder_components = [
        ("Input\nEmbedding", encoder_x, 8.5, embed_color),
        ("Positional\nEncoding", encoder_x, 7.5, embed_color),
        ("Multi-Head\nSelf-Attention", encoder_x, 6.0, attention_color),
        ("Add & Norm", encoder_x, 4.8, norm_color),
        ("Feed\nForward", encoder_x, 3.6, ff_color),
        ("Add & Norm", encoder_x, 2.4, norm_color)
    ]
    
    # Decoder components
    decoder_x = 7.0
    decoder_components = [
        ("Output\nEmbedding", decoder_x, 8.5, embed_color),
        ("Positional\nEncoding", decoder_x, 7.5, embed_color),
        ("Masked\nSelf-Attention", decoder_x, 6.0, attention_color),
        ("Add & Norm", decoder_x, 5.2, norm_color),
        ("Cross-\nAttention", decoder_x, 4.4, attention_color),
        ("Add & Norm", decoder_x, 3.6, norm_color),
        ("Feed\nForward", decoder_x, 2.8, ff_color),
        ("Add & Norm", decoder_x, 2.0, norm_color)
    ]
    
    # Draw encoder components
    encoder_patches = []
    for text, x, y, color in encoder_components:
        rect = patches.FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=color, alpha=0.8,
                                     edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        encoder_patches.append(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    
    # Draw decoder components
    decoder_patches = []
    for text, x, y, color in decoder_components:
        rect = patches.FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=color, alpha=0.8,
                                     edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        decoder_patches.append(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    
    # Final layers
    final_x = decoder_x
    final_y = 1.0
    final_components = [
        ("Linear", final_x, final_y, final_color),
        ("Softmax", final_x, final_y-1.0, final_color)
    ]
    
    for text, x, y, color in final_components:
        rect = patches.FancyBboxPatch((x-1.0, y-0.3), 2.0, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor=color, alpha=0.8,
                                     edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    
    # Draw arrows for encoder
    for i in range(len(encoder_components)-1):
        x1, y1 = encoder_x, encoder_components[i][2]-0.4
        x2, y2 = encoder_x, encoder_components[i+1][2]+0.4
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Draw arrows for decoder
    for i in range(len(decoder_components)-1):
        x1, y1 = decoder_x, decoder_components[i][2]-0.4
        x2, y2 = decoder_x, decoder_components[i+1][2]+0.4
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Arrow from last decoder component to Linear
    ax.annotate('', xy=(decoder_x, decoder_components[-1][2]-0.4),
               xytext=(decoder_x, final_y+0.3),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Arrow from Linear to Softmax
    ax.annotate('', xy=(final_x, final_y-0.3),
               xytext=(final_x, final_y+0.3),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Cross-attention connection
    encoder_attn_y = encoder_components[2][2]  # Multi-Head Self-Attention
    decoder_attn_y = decoder_components[4][2]  # Cross-Attention
    
    # Draw curved arrow from encoder to decoder
    ax.annotate('', xy=(decoder_x-1.2, decoder_attn_y),
               xytext=(encoder_x+1.2, encoder_attn_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='red',
                             connectionstyle="arc3,rad=-0.3"))
    
    # Add K, V label
    mid_x = (encoder_x + decoder_x) / 2
    mid_y = (encoder_attn_y + decoder_attn_y) / 2 - 0.2
    ax.text(mid_x, mid_y, 'K, V', ha='center', va='center',
            fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # Add titles
    ax.text(encoder_x, 9.5, 'ENCODER', ha='center', va='center',
            fontsize=14, fontweight='bold', color=encoder_color)
    ax.text(decoder_x, 9.5, 'DECODER', ha='center', va='center',
            fontsize=14, fontweight='bold', color=decoder_color)
    
    # Add input/output labels
    ax.text(encoder_x, 9.0, 'Input', ha='center', va='center',
            fontsize=10, fontweight='bold', color='black')
    ax.text(decoder_x, 9.0, 'Output', ha='center', va='center',
            fontsize=10, fontweight='bold', color='black')
    ax.text(decoder_x, -0.2, 'Predictions', ha='center', va='center',
            fontsize=10, fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.savefig('transformer_architecture.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    draw_transformer()
