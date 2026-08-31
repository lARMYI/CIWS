/**
 * Desktop shell.
 *
 * This is a window around the local hub, not a second implementation of it. It
 * spawns the same `python -m ciws` server every other entry point uses, waits
 * for the health endpoint, and loads the UI. Quitting the window stops the
 * server, so there is no orphaned process holding your database open.
 *
 * The shell is optional. `scripts/start.sh` and the browser work identically —
 * this exists for people who want CIWS in the dock rather than in a tab.
 */

const { app, BrowserWindow, shell, dialog } = require('electron')
const { spawn } = require('node:child_process')
const path = require('node:path')
const http = require('node:http')

const PORT = Number(process.env.CIWS_PORT || 8787)
const HOST = '127.0.0.1'
const ROOT = path.resolve(__dirname, '..', '..')

let server = null
let window = null

function pythonExecutable() {
  if (process.env.CIWS_PYTHON) return process.env.CIWS_PYTHON
  return process.platform === 'win32'
    ? path.join(ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(ROOT, '.venv', 'bin', 'python')
}

function startServer() {
  server = spawn(pythonExecutable(), ['-m', 'ciws', '--port', String(PORT)], {
    cwd: ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  server.stdout.on('data', (d) => process.stdout.write(`[ciws] ${d}`))
  server.stderr.on('data', (d) => process.stderr.write(`[ciws] ${d}`))
  server.on('exit', (code) => {
    server = null
    if (code && code !== 0 && !app.isQuiting) {
      dialog.showErrorBox(
        'CIWS stopped',
        `The hub exited with code ${code}. Run scripts/setup.sh (or setup.bat) and try again.`,
      )
    }
  })
}

function waitForHealth(attempts = 90) {
  return new Promise((resolve, reject) => {
    const probe = (left) => {
      const request = http.get(
        { host: HOST, port: PORT, path: '/api/health', timeout: 1200 },
        (response) => {
          response.resume()
          response.statusCode === 200 ? resolve() : retry(left)
        },
      )
      request.on('error', () => retry(left))
      request.on('timeout', () => {
        request.destroy()
        retry(left)
      })
    }
    const retry = (left) =>
      left <= 0 ? reject(new Error('the hub did not come up')) : setTimeout(() => probe(left - 1), 400)
    probe(attempts)
  })
}

async function createWindow() {
  window = new BrowserWindow({
    width: 1560,
    height: 960,
    minWidth: 1040,
    minHeight: 640,
    backgroundColor: '#05070a',
    title: 'CIWS',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    // Nothing from the page needs Node, so it does not get it.
    webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true },
    show: false,
  })

  // Links to the outside world open in the real browser, not inside the shell.
  window.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  try {
    await waitForHealth()
    await window.loadURL(`http://${HOST}:${PORT}/`)
  } catch (error) {
    await window.loadURL(
      'data:text/html,' +
        encodeURIComponent(
          `<body style="background:#05070a;color:#8296ad;font:13px ui-monospace;padding:40px">
             <h2 style="color:#22d3ee">CIWS could not start</h2>
             <p>${String(error.message)}</p>
             <p>Run <code style="color:#7dd3fc">scripts/setup.sh</code> (or <code>setup.bat</code>) from the project root, then reopen this app.</p>
           </body>`,
        ),
    )
  }
  window.show()
}

app.whenReady().then(() => {
  startServer()
  createWindow()
  app.on('activate', () => BrowserWindow.getAllWindows().length === 0 && createWindow())
})

app.on('before-quit', () => {
  app.isQuiting = true
  if (server) server.kill('SIGTERM')
})

app.on('window-all-closed', () => process.platform !== 'darwin' && app.quit())
