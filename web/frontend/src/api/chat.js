/**
 * Chat API — SSE streaming and REST helpers.
 */

const API_BASE = '/api'

export async function listSessions() {
  const res = await fetch(`${API_BASE}/sessions`)
  if (!res.ok) return { sessions: [] }
  return res.json()
}

export async function createSession() {
  const res = await fetch(`${API_BASE}/sessions`, { method: 'POST' })
  if (!res.ok) throw new Error('Failed to create session')
  return res.json()
}

export async function deleteSession(threadId) {
  const res = await fetch(`${API_BASE}/sessions/${threadId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete session')
  return res.json()
}

export async function renameSession(threadId, name) {
  const res = await fetch(`${API_BASE}/sessions/${threadId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!res.ok) throw new Error('Failed to rename session')
  return res.json()
}

export async function getHistory(threadId) {
  const res = await fetch(`${API_BASE}/sessions/${threadId}/history`)
  if (!res.ok) return { messages: [] }
  return res.json()
}

export async function getPlans(threadId) {
  const res = await fetch(`${API_BASE}/sessions/${threadId}/plans`)
  if (!res.ok) return { plans: [] }
  return res.json()
}

export async function listConstraints(threadId) {
  const res = await fetch(`${API_BASE}/sessions/${threadId}/constraints`)
  if (!res.ok) return { constraints: [] }
  return res.json()
}

export async function addConstraint(threadId, content) {
  const res = await fetch(`${API_BASE}/sessions/${threadId}/constraints`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  })
  if (!res.ok) throw new Error('Failed to add constraint')
  return res.json()
}

export async function toggleConstraint(threadId, constraintId, enabled) {
  const res = await fetch(`${API_BASE}/sessions/${threadId}/constraints/${constraintId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  })
  if (!res.ok) throw new Error('Failed to toggle constraint')
  return res.json()
}

export async function deleteConstraint(threadId, constraintId) {
  const res = await fetch(`${API_BASE}/sessions/${threadId}/constraints/${constraintId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to delete constraint')
  return res.json()
}

export async function confirmSql(sessionId, action, reason = '') {
  const res = await fetch(`${API_BASE}/chat/confirm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, action, reason }),
  })
  return res.json()
}

/**
 * Stream a query via SSE.
 *
 * @param {string} query
 * @param {string} threadId
 * @param {string} sessionId
 * @param {object} callbacks
 *   onNodeStart(node, label)
 *   onNodeEnd(node)
 *   onToken(content)
 *   onFullResponse(content)
 *   onSqlStep(stepId, label, sql, performance, optimization, elapsedMs)   — fired for each SQL step executed
 *   onSqlConfirm(sql, sessionId) → Promise<{action, reason}>
 *   onDone()
 *   onError(message)
 */
export function streamQuery(query, threadId, sessionId, callbacks) {
  const url = `${API_BASE}/chat/stream?query=${encodeURIComponent(query)}&thread_id=${encodeURIComponent(threadId)}&session_id=${encodeURIComponent(sessionId)}`
  const es = new EventSource(url)

  es.onmessage = async (e) => {
    let data
    try {
      data = JSON.parse(e.data)
    } catch {
      return
    }

    switch (data.type) {
      case 'node_start':
        callbacks.onNodeStart?.(data.node, data.label)
        break
      case 'node_end':
        callbacks.onNodeEnd?.(data.node)
        break
      case 'token':
        callbacks.onToken?.(data.content)
        break
      case 'full_response':
        callbacks.onFullResponse?.(data.content)
        break
      case 'sql_confirm': {
        // Pause SSE consumption while waiting for user confirmation
        const result = await callbacks.onSqlConfirm?.(data.sql, data.session_id)
        if (result) {
          await confirmSql(data.session_id, result.action, result.reason)
        }
        break
      }
      case 'sql_step':
        callbacks.onSqlStep?.(data.step_id, data.label, data.sql, data.performance, data.optimization, data.elapsed_ms)
        break
      case 'done':
        callbacks.onDone?.()
        es.close()
        break
      case 'error':
        callbacks.onError?.(data.message)
        es.close()
        break
    }
  }

  es.onerror = (e) => {
    callbacks.onError?.('SSE connection error')
    es.close()
  }

  return () => es.close()
}
