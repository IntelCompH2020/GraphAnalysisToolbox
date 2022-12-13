import os
import threading
# import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler


# from https://stackoverflow.com/a/38943044/3967334
def simple_http_server(host: str = '0.0.0.0', port: int = 4001, path: str = '.'):
	"""
	Sets up a http server for files.

	Parameters
	----------
	host: str
		The IP where the host will "listen"
	port: int
		The port where the host will "listen"
	path: str
		The path to be served

	Returns
	-------
	start: function
		It starts the server
	stop: function
		It stops the server

	"""

	server = HTTPServer((host, port), SimpleHTTPRequestHandler)
	thread = threading.Thread(target=server.serve_forever)
	thread.daemon = True

	cwd = os.getcwd()

	def start():
		os.chdir(path)
		thread.start()
		# webbrowser.open_new_tab('http://{}:{}'.format(host, port))
		print('starting server on port {}'.format(server.server_port))

	def stop():
		os.chdir(cwd)
		server.shutdown()
		server.socket.close()
		print('stopping server on port {}'.format(server.server_port))

	return start, stop
