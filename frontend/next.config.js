/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  
  // Remove static export for development
  // output: 'export',
  
  // Allow cross-origin requests in development
  allowedDevOrigins: ['http://192.168.1.2:3000', 'http://localhost:3000'],
  
  // Image configuration
  images: {
    unoptimized: process.env.NODE_ENV === 'production',
  },
  
  // Environment variables
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000',
  },
  
  // Production configuration for GitHub Pages
  basePath: process.env.NODE_ENV === 'production' ? '/2k_spark' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/2k_spark' : '',
  trailingSlash: false,
  // Turbopack configuration (now stable)
  turbo: {
    rules: {
      '*.svg': {
        loaders: ['@svgr/webpack'],
        as: '*.js',
      },
    },
    resolveAlias: {
      '@': './src',
    },
  },
};

module.exports = nextConfig;
