/**
 * Stats API — usage statistics endpoints.
 */

const API_BASE = '/api'

export async function getStatsOverview() {
  const res = await fetch(`${API_BASE}/stats/overview`)
  if (!res.ok) throw new Error('Failed to fetch stats overview')
  return res.json()
}

export async function getMonthlyStats(months = 12) {
  const res = await fetch(`${API_BASE}/stats/monthly?months=${months}`)
  if (!res.ok) throw new Error('Failed to fetch monthly stats')
  return res.json()
}
