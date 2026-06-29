import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Colors
    color_top = '#E8F5E9'
    color_mid = '#FFF3E0'
    color_bot = '#E3F2FD'
    color_edge = '#424242'
    color_box = '#F5F5F5'

    # 1. HTML DOM / Spatial Zoning (Left)
    # Draw a browser-like box
    browser_rect = patches.Rectangle((0.5, 1.5), 2.5, 4, linewidth=2, edgecolor=color_edge, facecolor='white', zorder=1)
    ax.add_patch(browser_rect)
    ax.text(1.75, 5.7, 'Webpage DOM', ha='center', va='center', fontsize=12, fontweight='bold', color=color_edge)

    # Top Zone
    top_rect = patches.Rectangle((0.6, 4.3), 2.3, 1.1, linewidth=1, edgecolor=color_edge, facecolor=color_top, zorder=2)
    ax.add_patch(top_rect)
    ax.text(1.75, 4.85, 'TOP ZONE', ha='center', va='center', fontsize=11, fontweight='bold', color='#2E7D32')
    ax.text(1.75, 4.5, '<html><body><header>...', ha='center', va='center', fontsize=8, color='#555555')

    # Middle Zone
    mid_rect = patches.Rectangle((0.6, 2.9), 2.3, 1.3, linewidth=1, edgecolor=color_edge, facecolor=color_mid, zorder=2)
    ax.add_patch(mid_rect)
    ax.text(1.75, 3.65, 'MIDDLE ZONE', ha='center', va='center', fontsize=11, fontweight='bold', color='#E65100')
    ax.text(1.75, 3.2, '<div><form><input>...', ha='center', va='center', fontsize=8, color='#555555')

    # Bottom Zone
    bot_rect = patches.Rectangle((0.6, 1.6), 2.3, 1.2, linewidth=1, edgecolor=color_edge, facecolor=color_bot, zorder=2)
    ax.add_patch(bot_rect)
    ax.text(1.75, 2.3, 'BOTTOM ZONE', ha='center', va='center', fontsize=11, fontweight='bold', color='#1565C0')
    ax.text(1.75, 1.9, '<footer><a>...', ha='center', va='center', fontsize=8, color='#555555')

    # 2. Arrows to XPath
    ax.annotate('', xy=(4.5, 3.5), xytext=(3.1, 4.8), arrowprops=dict(arrowstyle='->', lw=2, color=color_edge, connectionstyle="arc3,rad=0.2"))
    ax.annotate('', xy=(4.5, 3.5), xytext=(3.1, 3.5), arrowprops=dict(arrowstyle='->', lw=2, color=color_edge))
    ax.annotate('', xy=(4.5, 3.5), xytext=(3.1, 2.2), arrowprops=dict(arrowstyle='->', lw=2, color=color_edge, connectionstyle="arc3,rad=-0.2"))

    # 3. XPath Extraction Box
    xpath_rect = patches.FancyBboxPatch((4.5, 2.5), 2.5, 2, boxstyle="round,pad=0.1", linewidth=2, edgecolor=color_edge, facecolor=color_box)
    ax.add_patch(xpath_rect)
    ax.text(5.75, 4.1, '1. XPath Extraction', ha='center', va='center', fontsize=12, fontweight='bold', color=color_edge)
    ax.text(5.75, 3.5, 'html/body/div/form\\nhtml/body/footer/a\\nhtml/body/header/img', ha='center', va='center', fontsize=10, family='monospace', color='#333333')

    # 4. Arrow to TF-IDF
    ax.annotate('', xy=(8.5, 3.5), xytext=(7.1, 3.5), arrowprops=dict(arrowstyle='->', lw=2, color=color_edge))

    # 5. TF-IDF Box
    tfidf_rect = patches.FancyBboxPatch((8.5, 2.5), 2.5, 2, boxstyle="round,pad=0.1", linewidth=2, edgecolor=color_edge, facecolor=color_box)
    ax.add_patch(tfidf_rect)
    ax.text(9.75, 4.1, '2. Spatial TF-IDF', ha='center', va='center', fontsize=12, fontweight='bold', color=color_edge)
    ax.text(9.75, 3.5, 'Term Frequency -\\nInverse Document\\nFrequency on\\nXPath Sequences', ha='center', va='center', fontsize=10, color='#333333')

    # 6. Arrow down to final vector
    ax.annotate('', xy=(9.75, 1.5), xytext=(9.75, 2.4), arrowprops=dict(arrowstyle='->', lw=2, color=color_edge))

    # 7. Final Output Vector
    out_rect = patches.Rectangle((7.75, 0.5), 4.0, 1, linewidth=2, edgecolor='#1565C0', facecolor='#E3F2FD')
    ax.add_patch(out_rect)
    ax.text(9.75, 1.0, '163-Dimensional\nStructural Feature Vector', ha='center', va='center', fontsize=12, fontweight='bold', color='#1565C0')

    plt.savefig('d:/Desktop/PhishingDetection/xpath_spatial_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_diagram()
