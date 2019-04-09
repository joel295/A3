import server_helper as helper
import graph as g

db = helper.db

def create_graphs(db):
    # Create graphs and save files
    # Chest pain type
    g3 = g.graph(3, db)
    g3.create_plot_3(db)

    # Resting Blood Pressure
    g4 = g.graph(4, db)
    g4.create_plot_4(db)

    # Serum Cholestrol
    g5 = g.graph(5, db)
    g5.create_plot_5(db)

    # Fasting Blood Sugar
    g6 = g.graph(6, db)
    g6.create_plot_6(db)

    # Resting ECG
    g7 = g.graph(7, db)
    g7.create_plot_7(db)

    # Max Heart Rate
    g8 = g.graph(8, db)
    g8.create_plot_8(db)

    # Exercise Induced Angina
    g9 = g.graph(9, db)
    g9.create_plot_9(db)

    # Oldpeak


    # Slope
    g11 = g.graph(11,db)
    g11.create_plot_11(db)

    # Number Vessels Coloured by Flouroscopy
    g12 = g.graph(12,db)
    g12.create_plot_12(db)

    # Thalassemia
    g13 = g.graph(13, db)
    g13.create_plot_13(db)

create_graphs(db)