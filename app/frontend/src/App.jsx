import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

const BACKEND_URL = 'ws://localhost:8000/ws/emotion'
const PICTOGRAM_URL = 'http://localhost:8000/pictograms'

// Logo icono de cara/emoticono
const Logo = () => (
  <svg viewBox="0 0 100 100" className="logo-icon">
    <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="4" />
    <circle cx="35" cy="40" r="6" fill="currentColor" />
    <circle cx="65" cy="40" r="6" fill="currentColor" />
    <path d="M 30 65 Q 50 80 70 65" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
  </svg>
)

function App() {
  const [started, setStarted] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [emotion, setEmotion] = useState(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState(null)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const streamRef = useRef(null)
  const intervalRef = useRef(null)

  const emotionLabels = {
    angry: 'Enojo',
    disgust: 'Asco',
    fear: 'Miedo',
    happy: 'Felicidad',
    neutral: 'Neutral',
    sad: 'Tristeza',
    surprise: 'Sorpresa'
  }

  const startCamera = useCallback(async () => {
    try {
      setError(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 }
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }

      wsRef.current = new WebSocket(BACKEND_URL)

      wsRef.current.onopen = () => {
        setConnected(true)
        setIsRunning(true)
        startCapturing()
      }

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (data.detected) {
          setEmotion(data.emotion)
        } else {
          setEmotion(null)
        }
      }

      wsRef.current.onerror = () => {
        setError('Error de conexión')
        setConnected(false)
      }

      wsRef.current.onclose = () => {
        setConnected(false)
      }

    } catch (err) {
      setError('No se pudo acceder a la cámara')
    }
  }, [])

  const startCapturing = useCallback(() => {
    intervalRef.current = setInterval(() => {
      if (videoRef.current && canvasRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
        const canvas = canvasRef.current
        const video = videoRef.current
        const ctx = canvas.getContext('2d')

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx.drawImage(video, 0, 0)

        const frameData = canvas.toDataURL('image/jpeg', 0.6)
        wsRef.current.send(frameData)
      }
    }, 100)
  }, [])

  const stopCamera = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (wsRef.current) wsRef.current.close()
    if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop())
    setIsRunning(false)
    setConnected(false)
    setEmotion(null)
  }, [])

  const handleStart = () => {
    setStarted(true)
    startCamera()
  }

  const handleStop = () => {
    stopCamera()
    setStarted(false)
  }

  useEffect(() => {
    return () => stopCamera()
  }, [stopCamera])

  // Pantalla de bienvenida
  if (!started) {
    return (
      <div className="app">
        <div className="bg-animation">
          <div className="blob blob-1"></div>
          <div className="blob blob-2"></div>
          <div className="blob blob-3"></div>
        </div>
        <div className="welcome-screen">
          <div className="welcome-content">
            <div className="welcome-logo">
              <Logo />
            </div>
            <h1>MoodLens</h1>
            <p className="welcome-subtitle">
              Traductor de emociones a pictogramas TEA
            </p>
            <button onClick={handleStart} className="btn-start">
              Comenzar
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Pantalla principal
  return (
    <div className="app">
      <div className="bg-animation">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>
      <nav className="navbar">
        <div className="logo-nav">
          <Logo />
          <span>MoodLens</span>
        </div>
        <div className="nav-actions">
          <span className={`status-dot ${connected ? 'active' : ''}`}></span>
          <button onClick={handleStop} className="btn-back">
            Salir
          </button>
        </div>
      </nav>

      <main className="main">
        <div className="content">
          <div className="camera-section">
            <div className="camera-wrapper">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={isRunning ? 'active' : ''}
              />
              <canvas ref={canvasRef} style={{ display: 'none' }} />
            </div>
            {error && <p className="error-text">{error}</p>}
          </div>

          <div className="result-section">
            {emotion ? (
              <div className="emotion-result">
                <div className="pictogram-box">
                  <img
                    src={`${PICTOGRAM_URL}/${emotion}`}
                    alt={emotionLabels[emotion]}
                  />
                </div>
                <h2 className="emotion-name">{emotionLabels[emotion]}</h2>
              </div>
            ) : (
              <div className="no-result">
                <p>Buscando rostro...</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
