from flask import Flask, render_template, request, abort
from server_helper import graph_dict, test_list
import io

def server():
    app = Flask(__name__)
    app.debug =  True
    app.host='localhost'
    app.config['TESTING']
    return app

server = server()

@server.errorhandler(404)
def page_not_found(e):
    return render_template("bad.html"), 404

@server.route('/')
def home_page():
    return render_template("home.html")

@server.route('/graph')
@server.route('/graph<int:number>')
def graph(number=None):
    if number == None:
        return render_template("graphw.html")
    try:
        number = int(number)
        graph_name = graph_dict[number]
    except (ValueError, KeyError):
        abort(404)
    if not 1 <= number <= 14:
        abort(404)
    else:
        return render_template("graph.html", graph_name = graph_name)

@server.route('/factors')
def factors():
    # Look just edit this function in helper and you can build your table
    # Ensure the headers defined in factors.html are set if more columns added
    table = test_list()
    return render_template("factors.html", table=table)

@server.route('/predictor')
def predictor():
    return render_template("predictor.html")

@server.route('/other')
def other():
    return render_template("other.html")

server.run()