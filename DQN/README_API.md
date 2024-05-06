## Demonstrating the API

Open two terminals, one to run the API and another to access the API. If you are using a virtual environment make sure to activate it in both terminals. In one terminal execute the following:
```
python api_server.py
```
Executing the code snippet will produce the following output.
```
 * Serving Flask app 'api_server'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 622-015-947
```

Copy the http address, the API is running on, to clipboard. Open `v5_demo_api_test.py` and edit the following line to match the copied URL.
```python
base_url = 'http://127.0.0.1:5000'
```

You can then proceed to execute the following command in the second terminal.
```
python api_client.py
```
