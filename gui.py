import tkinter as tk
from tkinter import ttk

class GraphSelector:
    def __init__(self, strategies, benchmarks, global_plots, per_strategy_plots):
        self.root = tk.Tk()
        self.root.title("Institutional Backtester - Graph Selection")
        self.root.geometry("1000x800")
        
        self.strategies = strategies
        self.benchmarks = benchmarks
        self.all_entities = strategies + benchmarks
        self.global_plots = global_plots
        self.per_strategy_plots = per_strategy_plots
        
        self.selection = {
            'global': {plot: tk.BooleanVar(value=True) for plot in global_plots},
            'per_entity': {entity: {plot: tk.BooleanVar(value=True) for plot in per_strategy_plots} for entity in self.all_entities}
        }
        
        self._setup_ui()
        
    def _setup_ui(self):
        # Main container with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons Frame (Fixed at top)
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(btn_frame, text="Select All", command=self._select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Deselect All", command=self._deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="RENDER", command=self._on_render).pack(side=tk.RIGHT, padx=5)
        
        # 0. TDA Configuration Section
        tda_frame = ttk.LabelFrame(self.scrollable_frame, text="Topological Data Analysis (TDA)")
        tda_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.tda_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(tda_frame, text="Enable TDA Interactive Explorer", variable=self.tda_enabled).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)
        
        ttk.Label(tda_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
        self.tda_start = tk.StringVar(value="2020-02-18")
        ttk.Entry(tda_frame, textvariable=self.tda_start).grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
        
        ttk.Label(tda_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=2)
        self.tda_end = tk.StringVar(value="2020-04-06")
        ttk.Entry(tda_frame, textvariable=self.tda_end).grid(row=2, column=1, sticky=tk.W, padx=10, pady=2)
        
        ttk.Label(tda_frame, text="Window (Months):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=2)
        self.tda_window = tk.StringVar(value="6")
        ttk.Entry(tda_frame, textvariable=self.tda_window).grid(row=3, column=1, sticky=tk.W, padx=10, pady=2)

        # 1. Global Plots Section
        global_frame = ttk.LabelFrame(self.scrollable_frame, text="Global Plots")
        global_frame.pack(fill=tk.X, padx=5, pady=5)
        
        for i, plot in enumerate(self.global_plots):
            cb = ttk.Checkbutton(global_frame, text=plot, variable=self.selection['global'][plot])
            cb.grid(row=i // 3, column=i % 3, sticky=tk.W, padx=10, pady=2)
            
        # 2. Per-Strategy/Benchmark Grid
        grid_frame = ttk.LabelFrame(self.scrollable_frame, text="Per-Strategy / Benchmark Plots")
        grid_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Header Row
        ttk.Label(grid_frame, text="Entity", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        for j, plot in enumerate(self.per_strategy_plots):
            ttk.Label(grid_frame, text=plot, font=('Helvetica', 8, 'bold')).grid(row=0, column=j+1, padx=5, pady=5)
            
        # Entity Rows
        for i, entity in enumerate(self.all_entities):
            ttk.Label(grid_frame, text=entity).grid(row=i+1, column=0, sticky=tk.W, padx=5, pady=2)
            for j, plot in enumerate(self.per_strategy_plots):
                cb = ttk.Checkbutton(grid_frame, variable=self.selection['per_entity'][entity][plot])
                cb.grid(row=i+1, column=j+1, padx=5, pady=2)

    def _select_all(self):
        for var in self.selection['global'].values():
            var.set(True)
        for entity_plots in self.selection['per_entity'].values():
            for var in entity_plots.values():
                var.set(True)

    def _deselect_all(self):
        for var in self.selection['global'].values():
            var.set(False)
        for entity_plots in self.selection['per_entity'].values():
            for var in entity_plots.values():
                var.set(False)

    def _on_render(self):
        self.final_selection = {
            'global': {plot: var.get() for plot, var in self.selection['global'].items()},
            'per_entity': {entity: {plot: var.get() for plot, var in entity_plots.items()} for entity, entity_plots in self.selection['per_entity'].items()},
            'tda': {
                'enabled': self.tda_enabled.get(),
                'start': self.tda_start.get(),
                'end': self.tda_end.get(),
                'window': int(self.tda_window.get() if self.tda_window.get().isdigit() else 6)
            }
        }
        self.root.destroy()

    def get_selection(self):
        self.root.mainloop()
        return getattr(self, 'final_selection', None)
