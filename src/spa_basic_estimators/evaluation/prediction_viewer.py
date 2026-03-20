from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover - import guard for local runtime only
    raise SystemExit(
        "matplotlib is required for the prediction viewer. Install project dependencies first."
    ) from exc

import pandas as pd

from spa_basic_estimators.evaluation.prediction_store import (
    PredictionStoreInfo,
    choose_x_axis_column,
    compute_run_rmse,
    default_selected_columns,
    discover_prediction_stores,
    list_plottable_columns,
    load_run_catalog,
    load_run_frame,
)


class PredictionViewerApp:
    def __init__(self, root: tk.Tk, outputs_dir: str | Path = "outputs") -> None:
        self.root = root
        self.outputs_dir = Path(outputs_dir)
        self.root.title("SPA Prediction Viewer")
        self.root.geometry("1200x800")

        self.model_var = tk.StringVar()
        self.run_var = tk.StringVar()
        self.rmse_var = tk.StringVar(value="Run RMSE: -")
        self.split_var = tk.StringVar(value="Split: -")
        self.rows_var = tk.StringVar(value="Rows: -")
        self.x_axis_var = tk.StringVar(value="X-axis: -")

        self.store_options: dict[str, PredictionStoreInfo] = {}
        self.current_store: PredictionStoreInfo | None = None
        self.current_catalog = pd.DataFrame()
        self.current_frame = pd.DataFrame()

        self._build_layout()
        self._load_store_options()

    def _build_layout(self) -> None:
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Model").grid(row=0, column=0, sticky="w")
        self.model_combo = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            state="readonly",
            width=40,
        )
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(8, 16))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        ttk.Label(controls, text="Trial / Run").grid(row=0, column=2, sticky="w")
        self.run_combo = ttk.Combobox(
            controls,
            textvariable=self.run_var,
            state="readonly",
            width=40,
        )
        self.run_combo.grid(row=0, column=3, sticky="ew", padx=(8, 16))
        self.run_combo.bind("<<ComboboxSelected>>", self._on_run_changed)

        refresh_button = ttk.Button(controls, text="Refresh Plot", command=self.refresh_plot)
        refresh_button.grid(row=0, column=4, sticky="e")

        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        body = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(body)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        ttk.Label(left_panel, text="Variables").pack(anchor="w")
        self.variable_listbox = tk.Listbox(
            left_panel,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            width=28,
            height=20,
        )
        self.variable_listbox.pack(fill=tk.Y, expand=False, pady=(6, 12))
        self.variable_listbox.bind("<<ListboxSelect>>", self._on_variables_changed)

        ttk.Label(left_panel, textvariable=self.rmse_var).pack(anchor="w", pady=(0, 6))
        ttk.Label(left_panel, textvariable=self.split_var).pack(anchor="w", pady=(0, 6))
        ttk.Label(left_panel, textvariable=self.rows_var).pack(anchor="w", pady=(0, 6))
        ttk.Label(left_panel, textvariable=self.x_axis_var).pack(anchor="w", pady=(0, 6))

        plot_panel = ttk.Frame(body)
        plot_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Select a model and run to begin")
        self.axes.set_xlabel("Time")
        self.axes.set_ylabel("Value")
        self.axes.grid(alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_panel, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

    def _load_store_options(self) -> None:
        store_infos = discover_prediction_stores(self.outputs_dir)
        if not store_infos:
            raise FileNotFoundError(
                f"No prediction stores were found under '{self.outputs_dir}'. "
                "Run a model first so it writes all_dataset_predictions.h5."
            )

        self.store_options = {}
        for store_info in store_infos:
            display_name = self._build_store_display_name(store_info)
            self.store_options[display_name] = store_info

        self.model_combo["values"] = list(self.store_options)
        first_label = next(iter(self.store_options))
        self.model_var.set(first_label)
        self._on_model_changed()

    def _build_store_display_name(self, store_info: PredictionStoreInfo) -> str:
        if store_info.model_name == store_info.artifact_dir.name:
            return store_info.model_name
        return f"{store_info.model_name} [{store_info.artifact_dir.name}]"

    def _on_model_changed(self, event: object | None = None) -> None:
        selection = self.model_var.get()
        if not selection:
            return

        self.current_store = self.store_options[selection]
        self.current_catalog = load_run_catalog(self.current_store.store_path)
        run_ids = self.current_catalog["run_id"].astype(str).tolist()
        self.run_combo["values"] = run_ids

        if not run_ids:
            self.run_var.set("")
            self.current_frame = pd.DataFrame()
            self._clear_plot("Selected model does not contain any saved runs")
            return

        self.run_var.set(run_ids[0])
        self._on_run_changed()

    def _on_run_changed(self, event: object | None = None) -> None:
        if self.current_store is None or not self.run_var.get():
            return

        self.current_frame = load_run_frame(self.current_store.store_path, self.run_var.get())
        self._update_run_controls()
        self.refresh_plot()

    def _update_run_controls(self) -> None:
        if self.current_frame.empty:
            self._populate_variable_list([])
            self.rmse_var.set("Run RMSE: -")
            self.split_var.set("Split: -")
            self.rows_var.set("Rows: -")
            self.x_axis_var.set("X-axis: -")
            return

        split_label = "-"
        if "split" in self.current_frame.columns and not self.current_frame["split"].empty:
            split_label = str(self.current_frame["split"].iloc[0])

        self.rmse_var.set(f"Run RMSE: {compute_run_rmse(self.current_frame):.6f}")
        self.split_var.set(f"Split: {split_label}")
        self.rows_var.set(f"Rows: {len(self.current_frame)}")
        self.x_axis_var.set(f"X-axis: {choose_x_axis_column(self.current_frame)}")

        x_axis_column = choose_x_axis_column(self.current_frame)
        available_columns = [
            column for column in list_plottable_columns(self.current_frame) if column != x_axis_column
        ]
        self._populate_variable_list(available_columns)
        default_columns = [
            column for column in default_selected_columns(self.current_frame) if column != x_axis_column
        ]
        self._select_variables(default_columns)

    def _populate_variable_list(self, columns: list[str]) -> None:
        self.variable_listbox.delete(0, tk.END)
        for column in columns:
            self.variable_listbox.insert(tk.END, column)

    def _select_variables(self, columns: list[str]) -> None:
        self.variable_listbox.selection_clear(0, tk.END)
        for index, candidate in enumerate(self.variable_listbox.get(0, tk.END)):
            if candidate in columns:
                self.variable_listbox.selection_set(index)

    def _selected_columns(self) -> list[str]:
        selected_indices = self.variable_listbox.curselection()
        return [self.variable_listbox.get(index) for index in selected_indices]

    def _on_variables_changed(self, event: object | None = None) -> None:
        self.refresh_plot()

    def refresh_plot(self) -> None:
        if self.current_frame.empty:
            self._clear_plot("No run selected")
            return

        selected_columns = self._selected_columns()
        if not selected_columns:
            self._clear_plot("Select at least one variable to plot")
            return

        x_axis_column = choose_x_axis_column(self.current_frame)
        x_values = self.current_frame[x_axis_column].to_numpy(dtype=float)

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        for column in selected_columns:
            self.axes.plot(x_values, self.current_frame[column].to_numpy(dtype=float), label=column)

        model_label = self.model_var.get()
        run_label = self.run_var.get()
        self.axes.set_title(f"{model_label} | {run_label}")
        self.axes.set_xlabel(x_axis_column)
        self.axes.set_ylabel("Value")
        self.axes.grid(alpha=0.3)
        self.axes.legend(loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _clear_plot(self, message: str) -> None:
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title(message)
        self.axes.set_xlabel("Time")
        self.axes.set_ylabel("Value")
        self.axes.grid(alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch an interactive viewer for saved per-run prediction stores."
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing model artifact folders with all_dataset_predictions.h5 files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = tk.Tk()
    try:
        PredictionViewerApp(root, outputs_dir=args.outputs_dir)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("Prediction Viewer", str(exc))
        root.destroy()
        raise SystemExit(str(exc)) from exc

    root.mainloop()


if __name__ == "__main__":
    main()
