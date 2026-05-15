import customtkinter as ctk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog

from tecnicas_de_balanceo import balanceo_smote, balanceo_adasyn, balanceo_rus
from tecnicas_de_estandarizacion import min_max, z_score, robust_scaler, max_abs, power_transformer

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import numpy as np

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Proyecto Final")
        self.geometry("1200x800")

        self.df = None
        self.m = None
        self.y = None
        self.target_col = None

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        self.tabview.add("Carga de datos")
        self.tabview.add("Preprocesamiento")
        self.tabview.add("Modelado")
        self.tabview.add("Resultados")

        # -------- CARGA --------
        self.btn_csv = ctk.CTkButton(
            self.tabview.tab("Carga de datos"),
            text="Leer CSV",
            command=self.leer_csv
        )
        self.btn_csv.pack(pady=10)

        self.status_label = ctk.CTkLabel(
            self.tabview.tab("Carga de datos"),
            text="Esperando acción..."
        )
        self.status_label.pack(pady=10)

        self.graph_frame = ctk.CTkFrame(self.tabview.tab("Carga de datos"))
        self.graph_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # -------- PREPROCESAMIENTO --------
        ctk.CTkLabel(self.tabview.tab("Preprocesamiento"), text="Balanceo de clases").pack(pady=5)

        self.balanceo_menu = ctk.CTkOptionMenu(
            self.tabview.tab("Preprocesamiento"),
            values=["SMOTE", "ADASYN", "RUS"],
            command=self.balancear
        )
        self.balanceo_menu.pack(pady=10)

        ctk.CTkLabel(self.tabview.tab("Preprocesamiento"), text="Escalamiento de datos").pack(pady=5)

        self.escalamiento_menu = ctk.CTkOptionMenu(
            self.tabview.tab("Preprocesamiento"),
            values=["Min-Max", "Z-score", "Robust", "Max-abs", "Power"],
            command=self.escalar
        )
        self.escalamiento_menu.pack(pady=10)

        self.graph_frame_pre = ctk.CTkFrame(self.tabview.tab("Preprocesamiento"))
        self.graph_frame_pre.pack(fill="both", expand=True, padx=20, pady=20)

        # -------- MODELADO --------
        ctk.CTkLabel(self.tabview.tab("Modelado"), text="Selecciona clasificador").pack(pady=5)

        self.modelo_menu = ctk.CTkOptionMenu(
            self.tabview.tab("Modelado"),
            values=["Árbol de Decisión", "Random Forest", "KNN"],
            command=self.clasificar
        )
        self.modelo_menu.pack(pady=10)

        self.graph_frame_model = ctk.CTkFrame(self.tabview.tab("Modelado"))
        self.graph_frame_model.pack(fill="both", expand=True, padx=20, pady=20)

        # -------- RESULTADOS --------
        self.graph_frame_results = ctk.CTkFrame(self.tabview.tab("Resultados"))
        self.graph_frame_results.pack(fill="both", expand=True, padx=20, pady=20)

    # -------- MOSTRAR GRÁFICA --------
    def mostrar_grafica(self, fig, frame):
        for w in frame.winfo_children():
            w.destroy()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # -------- LEER CSV --------
    def leer_csv(self):
        ruta = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

        if ruta:
            self.df = pd.read_csv(ruta)
            self.status_label.configure(text="Archivo cargado correctamente")
            self.seleccionar_target()

    # -------- SELECCIONAR TARGET --------
    def seleccionar_target(self):
        ventana = ctk.CTkToplevel(self)
        ventana.geometry("300x200")
        ventana.title("Seleccionar target")

        ctk.CTkLabel(ventana, text="Columna objetivo").pack(pady=10)

        opciones = list(self.df.columns)
        var = ctk.StringVar(value=opciones[0])

        menu = ctk.CTkOptionMenu(ventana, values=opciones, variable=var)
        menu.pack(pady=10)

        def confirmar():
            self.target_col = var.get()

            self.m = self.df.drop(columns=[self.target_col])
            self.y = self.df[self.target_col]

            # 🔴 1. eliminar NaN en y
            mask = self.y.notna()
            self.m = self.m[mask]
            self.y = self.y[mask]

            # 2. convertir texto a numérico
            self.m = pd.get_dummies(self.m)

            # 3. rellenar NaN en X
            self.m = self.m.fillna(self.m.mean())

            fig, ax = plt.subplots()
            self.y.value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Distribución: {self.target_col}")

            self.mostrar_grafica(fig, self.graph_frame)
            plt.close(fig)

            ventana.destroy()

        ctk.CTkButton(ventana, text="Confirmar", command=confirmar).pack(pady=10)

    # -------- BALANCEO --------
    def balancear(self, metodo):
        if self.m is not None and self.y is not None:
            try:
                if metodo == "SMOTE":
                    self.m, self.y = balanceo_smote(self.m, self.y)
                elif metodo == "ADASYN":
                    self.m, self.y = balanceo_adasyn(self.m, self.y)
                elif metodo == "RUS":
                    self.m, self.y = balanceo_rus(self.m, self.y)

                fig, ax = plt.subplots()
                self.y.value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Después de {metodo}")

                self.mostrar_grafica(fig, self.graph_frame_pre)
                plt.close(fig)

            except Exception as e:
                print("Error en balanceo:", e)

    # -------- ESCALAR --------
    def escalar(self, metodo):
        if self.m is not None:
            try:
                if metodo == "Min-Max":
                    self.m = min_max(self.m)
                elif metodo == "Z-score":
                    self.m = z_score(self.m)
                elif metodo == "Robust":
                    self.m = robust_scaler(self.m)
                elif metodo == "Max-abs":
                    self.m = max_abs(self.m)
                elif metodo == "Power":
                    self.m = power_transformer(self.m)

                col = self.m.columns[0]

                fig, ax = plt.subplots()
                self.m[col].hist(ax=ax)
                ax.set_title(f"Escalado: {metodo}")

                self.mostrar_grafica(fig, self.graph_frame_pre)
                plt.close(fig)

            except Exception as e:
                print("Error en escalamiento:", e)

    # -------- CLASIFICAR --------
    def clasificar(self, metodo):
        if self.m is None or self.y is None:
            return

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.m, self.y, test_size=0.2, random_state=42
            )

            if metodo == "Árbol de Decisión":
                model = DecisionTreeClassifier()
            elif metodo == "Random Forest":
                model = RandomForestClassifier()
            else:
                model = KNeighborsClassifier()

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)

            # MATRIZ DE CONFUSIÓN
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, pred)
            ax.imshow(cm, cmap="Blues")
            ax.set_title(f"Accuracy: {acc:.2f}")
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Real")

            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            self.mostrar_grafica(fig, self.graph_frame_model)
            plt.close(fig)

            # REPORTE
            report = classification_report(y_test, pred, output_dict=True)

            clases = list(report.keys())[:-3]
            data = [
                [report[c]["precision"], report[c]["recall"], report[c]["f1-score"]]
                for c in clases
            ]

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.axis("off")

            ax.table(
                cellText=np.round(data, 2),
                rowLabels=clases,
                colLabels=["precision", "recall", "f1"],
                loc="center"
            )

            self.mostrar_grafica(fig, self.graph_frame_results)
            plt.close(fig)

        except Exception as e:
            print("Error en clasificación:", e)


if __name__ == "__main__":
    app = App()
    app.mainloop()
