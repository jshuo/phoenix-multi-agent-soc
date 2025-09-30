import './globals.css'

export const metadata = {
  title: 'Arviem-ITracXing AIoT Digital Twin - MaaS Platform',
  description: 'Multi-Agent Logistics Monitoring System',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}