# Local Windows run helper - uses Flask/Werkzeug dev server (reliable on Windows).
# Eventlet (used on Render/Linux) has known binding issues on Windows that can
# prevent socketio.run() from listening. This uses app.run() so pages render.
import os
os.environ['FLASK_DEBUG'] = '1'

import app as m

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    m.app.run(host='127.0.0.1', port=5000, debug=debug_mode, use_reloader=False, threaded=True)
