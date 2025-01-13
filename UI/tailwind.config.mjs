/** @type {import('tailwindcss').Config} */
export default {
    content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
    theme: {
      extend: {
        colors: {
          dark: {
            bg: '#121212',
            surface: '#1E1E1E',
            divider: '#272727',
          },
          accent: {
            primary: '#BB86FC',
            secondary: '#03DAC6',
            error: '#CF6679',
          },
          text: {
            primary: '#FFFFFF',
            secondary: '#B0B0B0',
          },
          status: {
            success: '#4CAF50',
            info: '#2196F3',
          },
        },
      },
    },
    plugins: [],
  }