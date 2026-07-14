import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DARK_BACKGROUND = "#030315"
LIGHT_TEXT = "white"
ACCENT_CYAN = "#62d9ff"
ACCENT_GREEN = "#55d98b"
ACCENT_MAGENTA = "#ff70d5"

class Formato:
    @staticmethod
    def apply_dark_axes_style(fig, ax, title, xlabel, ylabel):
        ax.set_facecolor(DARK_BACKGROUND)
        fig.patch.set_facecolor(DARK_BACKGROUND)

        ax.set_title(title, color=LIGHT_TEXT, pad=12, fontweight="bold")
        ax.set_xlabel(xlabel, color=LIGHT_TEXT)
        ax.set_ylabel(ylabel, color=LIGHT_TEXT)
        ax.tick_params(axis="both", colors=LIGHT_TEXT)

        for spine in ax.spines.values():
            spine.set_edgecolor(LIGHT_TEXT)

    @staticmethod
    def add_dark_colorbar(fig, ax, image, label="Frecuencia"):
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(label, color=LIGHT_TEXT)
        colorbar.ax.yaxis.set_tick_params(color=LIGHT_TEXT)
        plt.setp(colorbar.ax.yaxis.get_ticklabels(), color=LIGHT_TEXT)
        colorbar.outline.set_edgecolor(LIGHT_TEXT)
        return colorbar
    
    @staticmethod
    def save_fig(fig, filename, dpi=500):
        fig.savefig(
            filename,
            dpi =  dpi,
            bbox_inches = 'tight',
            facecolor = fig.get_facecolor()
        )