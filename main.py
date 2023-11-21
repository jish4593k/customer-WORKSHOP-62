import pickle
import requests
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


def generate_plot():
    # Function to generate a sample plot (you can replace it with your actual data and visualization)
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 15, 25, 30]

    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Sample Plot')
    
    # Save plot to a BytesIO object
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    
    # Convert the plot to a base64 encoded string
    img_str = base64.b64encode(img_data.read()).decode('utf-8')

    return img_str


@app.route('/')
def index():
    # Display the plot on the home page
    plot_img = generate_plot()
    return render_template('index.html', plot_img=plot_img)


@app.route('/predict', methods=['POST'])
def predict():
   
    income = float(request.form.get('income'))
    spendi = float(request.form.get('spend'))

    # Make prediction using the loaded model
    ans = model.predict([[income, spendi]])[0]

   
    labels = ['Careless', 'Standard', 'Target', 'Sensible', 'Careful']
    pred = labels[ans]

    # Map the prediction to a corresponding description
    descriptions = [
        'This Group is a Careless Group Because their spending is High and their Annual Income is Low.',
        'This Group is a Standard Group, Because their spending is Average and their Annual Income is Average.',
        'This Group is Our Target Group, Because their spending is High and their Annual Income is High.',
        'This Group is Our Sensible Group, Because their spending is Low and their Annual Income is Low.',
        'This Group is Our Careful Group, Because their spending is Low and their Annual Income is High.'
    ]
    description = descriptions[ans]

    return render_template('index.html', pred=pred, description=description)


if __name__ == '__main__':
    app.run(debug=True)
