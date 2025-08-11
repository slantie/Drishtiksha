/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "primary":{
          "main": "#f56565",
          "light": "#ff7070",
          "lighter": "#fc6f6f",
          "dark": "#",
          "darker": "#"
        },

        "dark": {
          "text": "#DFDFD6",
          "background": "#1B1B1F",
          "highlight": "#fc6f6f",
          "muted": {
            "text": "#FFFFFF",
            "background": "#202127",
          },
          "noisy":{
            "text": "#98989F",
            "background": "#161618",
          },
          "hover": "#414853",
          "secondary": "#32363F",
          "tertiary": "#98989F"
        },

        "light": {
          "text": "#3C3C43",
          "background": "#FFFFFF",
          "highlight": "#F56565",
          "muted": {
            "text": "#67676C",
            "background": "#F6F6F7",
          },
          "noisy":{
            "text": "#67676C",
            "background": "#C2C2C4",
          },
          "hover": "#E4E4E9",
          "secondary": "#EBEBEF",
          "tertiary": "#98989F"
        }

      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          '"Helvetica Neue"',
          "Arial",
          '"Noto Sans"',
          "sans-serif",
          '"Apple Color Emoji"',
          '"Segoe UI Emoji"',
          '"Segoe UI Symbol"',
          '"Noto Color Emoji"',
        ],
      },
    },
  },
  plugins: [],
};
