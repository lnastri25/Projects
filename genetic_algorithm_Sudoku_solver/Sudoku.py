import random
import time
import json
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from tkinter import ttk
import GA_Sudoku_Solver as gss

random.seed(time.time())

CSV_FILENAME = "results_log.csv"

# Reiniciar CSV al iniciar el script
if os.path.exists(CSV_FILENAME):
    os.remove(CSV_FILENAME)

with open(CSV_FILENAME, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Difficulty", "Population_Size", "Time_Elapsed_s", "Generations"])


class SudokuGUI(Frame):
    def __init__(self, master, file):
        Frame.__init__(self, master)
        if master:
            master.title("Sudoku Solver – Genetic Algorithm")
            master.configure(bg="#2e2e2e")

        self.grid = [[0 for _ in range(9)] for _ in range(9)]
        self.grid_2 = None
        self.locked = []
        self.easy, self.medium, self.hard, self.expert = [], [], [], []
        self.load_db(file)

        self.make_grid()

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Helvetica", 14), padding=6)
        style.configure("Hover.TRadiobutton",
                        font=("Helvetica", 13),
                        background="#2e2e2e",
                        foreground="white")
        style.map("Hover.TRadiobutton",
                  background=[("active", "#000000")],
                  foreground=[("active", "#ffffff")])

        self.bframe = Frame(self, bg="#2e2e2e")
        self.lvVar = StringVar(value="")

        Label(self.bframe, text="Select difficulty:", font=("Helvetica", 20, "bold"),
              fg="white", bg="#2e2e2e").pack(pady=(10, 5))

        radio_frame = Frame(self.bframe, bg="#2e2e2e")
        radio_frame.pack(pady=5)

        for level in ["Easy", "Medium", "Hard", "Expert"]:
            ttk.Radiobutton(radio_frame,
                            text=level,
                            variable=self.lvVar,
                            value=level,
                            style="Hover.TRadiobutton").pack(anchor='center', pady=2)

        # ENTRY para Population Size
        Label(self.bframe, text="Population Size:", font=("Helvetica", 13),
              fg="white", bg="#2e2e2e").pack()
        self.pop_var = StringVar(value="1000")
        Entry(self.bframe, textvariable=self.pop_var, font=("Helvetica", 12),
              width=10, justify="center").pack(pady=(0, 10))

        ttk.Button(self.bframe, text='Generate New Game', command=self.new_game).pack(pady=10)
        ttk.Button(self.bframe, text='Solve Puzzle', command=self.solver).pack(pady=5)

        self.result_log = Text(self.bframe, height=8, width=50, bg="#2e2e2e", fg="lightgreen",
                               font=("Courier", 12), borderwidth=0, highlightthickness=0, wrap="none")
        self.result_log.pack(pady=10)
        self.result_log.configure(state='disabled')

        self.bframe.pack(side='bottom', fill='x', expand=True)
        self.pack()

    def load_db(self, file):
        with open(file) as f:
            data = json.load(f)
        self.easy = data['Easy']
        self.medium = data['Medium']
        self.hard = data['Hard']
        self.expert = data['Expert']

    def new_game(self):
        level = self.lvVar.get()
        if level == "Easy":
            self.given = self.easy[random.randint(0, len(self.easy) - 1)]
        elif level == "Medium":
            self.given = self.medium[random.randint(0, len(self.medium) - 1)]
        elif level == "Hard":
            self.given = self.hard[random.randint(0, len(self.hard) - 1)]
        elif level == "Expert":
            self.given = self.expert[random.randint(0, len(self.expert) - 1)]
        else:
            self.given = "0" * 81

        self.grid = np.array([int(x) for x in self.given]).reshape((9, 9))
        self.grid_2 = None
        self.sync_board_and_canvas(initial=True)

    def solver(self):
        try:
            population_size = int(self.pop_var.get())
        except ValueError:
            population_size = 1000  # Valor por defecto si la entrada es inválida

        s = gss.Sudoku(population_size=population_size)
        s.load(self.grid)
        start_time = time.time()
        generation, solution = s.solve(update_callback=self.update_board_with_candidate)

        if solution:
            time_elapsed = float('{0:6.2f}'.format(time.time() - start_time))
            if generation == -1:
                msg = "Invalid input, please generate a new game"
            elif generation == -2:
                msg = "No solution found, please try again"
            else:
                self.grid_2 = solution.values
                self.sync_board_and_canvas(initial=False)
                difficulty = self.lvVar.get()
                msg = (f"✅ {difficulty} mode: solved in generation {generation}\n"
                      f"⏱ Time elapsed:  {time_elapsed} s\n"
                      f"👥 Population size used: {population_size}")
                self.save_result_to_csv(difficulty, population_size, time_elapsed, generation)
                self.generate_plot()

            self.result_log.configure(state='normal')
            self.result_log.insert(END, msg + "\n\n")
            self.result_log.configure(state='disabled')

    def save_result_to_csv(self, difficulty, population_size, time_elapsed, generation):
        with open(CSV_FILENAME, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([difficulty, population_size, time_elapsed, generation])

    def generate_plot(self):
        df = pd.read_csv(CSV_FILENAME)

        plt.figure(figsize=(10, 6))
        for difficulty in df['Difficulty'].unique():
            subset = df[df['Difficulty'] == difficulty].sort_values("Population_Size")
            plt.plot(subset["Population_Size"], subset["Time_Elapsed_s"],
                     marker='o', label=difficulty)

            for _, row in subset.iterrows():
                plt.annotate(f'{row["Time_Elapsed_s"]:.2f}',
                             (row["Population_Size"], row["Time_Elapsed_s"]),
                             textcoords="offset points", xytext=(0, 8),
                             ha='center', fontsize=8, color='gray')

        plt.xscale("linear")
        plt.yscale("log")
        plt.yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1000", "10000"])
        plt.xlabel("Initial Population Size")
        plt.ylabel("Time Elapsed (s)")
        plt.title("Time Elapsed (s) versus Initial Population Size")
        plt.legend(title="Difficulty Level")
        plt.grid(which="major", axis="y", linestyle="--", linewidth=0.5)
        plt.grid(which="minor", axis="both", linestyle="", linewidth=0)  # Desactiva grid menor
        plt.tight_layout()
        plt.savefig("performance_plot.png")
        plt.close()

    def make_grid(self):
        cell_size = 60
        canvas_width = cell_size * 9
        canvas_height = cell_size * 9
        c = Canvas(self, width=canvas_width, height=canvas_height, bg="#1e1e1e", highlightthickness=0)
        c.pack(side='top', fill='both', expand=True)

        self.canvas = c
        self.handles = [[None for _ in range(9)] for _ in range(9)]

        for y in range(9):
            for x in range(9):
                x1 = x * cell_size
                y1 = y * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                color = "#f4f4f4" if (x // 3 + y // 3) % 2 == 0 else "#dddddd"
                rect = c.create_rectangle(x1, y1, x2, y2, fill=color, outline="#888888", width=1)
                text = c.create_text(x1 + cell_size // 2, y1 + cell_size // 2,
                                     font=('Helvetica', 18, 'bold'), text="")
                self.handles[y][x] = (rect, text)

        for i in range(10):
            lw = 3 if i % 3 == 0 else 1
            c.create_line(0, i * cell_size, canvas_width, i * cell_size, fill="black", width=lw)
            c.create_line(i * cell_size, 0, i * cell_size, canvas_height, fill="black", width=lw)

    def sync_board_and_canvas(self, initial=True):
        source = self.grid if initial else self.grid_2
        g_original = self.grid

        for y in range(9):
            for x in range(9):
                val = source[y][x]
                if val != 0:
                    if initial or g_original[y][x] != 0:
                        color = "black"
                    else:
                        color = "green" if self.is_valid(source, y, x) else "red"
                    self.canvas.itemconfig(self.handles[y][x][1], text=str(val), fill=color)
                else:
                    self.canvas.itemconfig(self.handles[y][x][1], text="")

    def is_valid(self, grid, row, col):
        val = grid[row][col]
        if val == 0:
            return True

        for c in range(9):
            if c != col and grid[row][c] == val:
                return False
        for r in range(9):
            if r != row and grid[r][col] == val:
                return False
        sr, sc = 3 * (row // 3), 3 * (col // 3)
        for r in range(sr, sr + 3):
            for c in range(sc, sc + 3):
                if (r != row or c != col) and grid[r][c] == val:
                    return False
        return True

    def update_board_with_candidate(self, candidate):
        self.grid_2 = candidate.values
        self.sync_board_and_canvas(initial=False)
        self.update_idletasks()


#####
file = "Sudoku_database.json"
tk = Tk()
gui = SudokuGUI(tk, file)
gui.mainloop()