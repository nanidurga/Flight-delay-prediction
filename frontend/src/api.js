import axios from 'axios'

// In production (Vercel) VITE_API_URL = https://mtp-flight-api.onrender.com
// In local dev it is unset, so we fall back to the Vite proxy prefix '/api'
const http = axios.create({ baseURL: import.meta.env.VITE_API_URL ?? '/api' })

export const getModelInfo   = ()         => http.get('/model/info').then(r => r.data)
export const getOptions     = ()         => http.get('/meta/options').then(r => r.data)
export const getStats       = ()         => http.get('/stats/overview').then(r => r.data)
export const getLiveFlights = (limit=40) => http.get(`/flights/live?limit=${limit}`).then(r => r.data)
export const getWeather     = (iata)     => http.get(`/flights/weather?iata=${iata}`).then(r => r.data)

export const predictFlight = (payload) =>
  http.post('/predict', payload).then(r => r.data)
