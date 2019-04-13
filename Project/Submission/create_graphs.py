import server_helper as helper
import graph as g
import graph_labelled as gl

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
    g10 = g.graph(10, db)
    g10.create_plot_10(db)

    # Slope
    g11 = g.graph(11,db)
    g11.create_plot_11(db)

    # Number Vessels Coloured by Flouroscopy
    g12 = g.graph(12,db)
    g12.create_plot_12(db)

    # Thalassemia
    g13 = g.graph(13, db)
    g13.create_plot_13(db)

    ##########
    #by heart disease
    # Chest pain type
    g3_lab = gl.graph(3, db)
    g3_lab.create_plot_3_labelled(db)

    # Resting Blood Pressure
    g4_lab = gl.graph(4, db)
    g4_lab.create_plot_4_labelled(db)

    # Serum Cholestrol
    g5_lab = gl.graph(5, db)
    g5_lab.create_plot_5_labelled(db)

    # Fasting Blood Sugar
    #g6_lab = gl.graph(6, db)
    #g6_lab.create_plot_6(db)

    # Resting ECG
    g7_lab = gl.graph(7, db)
    g7_lab.create_plot_7_labelled(db)

    # Max Heart Rate
    g8_lab = gl.graph(8, db)
    g8_lab.create_plot_8_labelled(db)

    # Exercise Induced Angina
    #g9_lab = gl.graph(9, db)
    #g9_lab.create_plot_9(db)

    # Oldpeak
    g10_lab = gl.graph(10, db)
    g10_lab.create_plot_10_labelled(db)

    # Slope
    #g11_lab = gl.graph(11,db)
    #g11_lab.create_plot_11(db)

    # Number Vessels Coloured by Flouroscopy
    g12_lab = gl.graph(12,db)
    g12_lab.create_plot_12_labelled(db)

    # Thalassemia
    g13_lab = gl.graph(13, db)
    g13_lab.create_plot_13_labelled(db)

    g15_lab = gl.graph(15,db)
    g15_lab.create_plot_15_labelled(db)

    g17_lab = gl.graph(17,db)
    g17_lab.create_plot_17_labelled(db)



create_graphs(db)