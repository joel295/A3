from flask import Flask, render_template, request, abort
import server_helper as helper
import io

#startups pre front end setup
print(f' * Database located @ {helper.db.database_name}')

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(e):
    return render_template("bad.html"), 404

@app.route('/')
def home_page():
    return render_template("home.html")

@app.route('/graph')
@app.route('/graph<int:number>')
def graph(number=None):
    if number == None:
        return render_template("graphw.html")
    try:
        number = int(number)
        graph_name = helper.graph_dict[number]
    except (ValueError, KeyError):
        abort(404)
    if not 1 <= number <= 14:
        abort(404)
    else:
        # Note: ensure the image is saved in /static/images, simply set a function to return the name of the image
        # associated
        my_image = helper.image_name_finder(number)
        if not my_image:
            abort(404)
        else:
            return render_template("graph.html", graph_name = graph_name, image_name = my_image)

@app.route('/factors')
def factors():
    # Look just edit this function in helper and you can build your table
    # Ensure the headers defined in factors.html are set if more columns added
    table = helper.test_list()
    return render_template("factors.html", table=table)

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        result = request.form
        error = helper.validate_result(result)
        if error:
            return render_template("incorrect_input.html", error=error)
        else:
            prediction = helper.predicted(result)
            return render_template("predict_result.html", predict=10)#=prediction)
    return render_template("predictor.html")

@app.route('/other')
def other():
    return render_template("other.html")




