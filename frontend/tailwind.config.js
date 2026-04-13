/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        sky: {
          950: '#0a1628',
          900: '#0f2040',
        }
      }
    }
  },
  plugins: []
}
