from flask import Flask, request, jsonify   # Import necessary packages
# import flask, request, jsonify   # Import necessary packages

PORT = 5000    # Specify the port the server will respond to  
app = Flask(__name__)    # Define the app variable  
languages = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Swedish"]  

@app.route("/")        # Handle the site home address  
def home():  
    return jsonify({"status": "online"})    # Return site status of online  

@app.route("/languages")    # Return list of languages  
def get_languages():  
    return jsonify({"languages": languages})  

if __name__ == "__main__":    # Start the server or listener operation  
    app.run(debug=True, host="0.0.0.0", port=PORT)