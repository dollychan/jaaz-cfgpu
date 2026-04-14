import { sendMessages } from '@/api/chat'
import Blur from '@/components/common/Blur'
import { ScrollArea } from '@/components/ui/scroll-area'
import { eventBus, TEvents } from '@/lib/event'
import ChatMagicGenerator from './ChatMagicGenerator'
import {
  AssistantMessage,
  Message,
  Model,
  PendingType,
  Session,
} from '@/types/types'
import { useSearch } from '@tanstack/react-router'
import { produce } from 'immer'
import { motion } from 'motion/react'
import { nanoid } from 'nanoid'
import {
  Dispatch,
  SetStateAction,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import { useTranslation } from 'react-i18next'
import { PhotoProvider } from 'react-photo-view'
import { toast } from 'sonner'
import ShinyText from '../ui/shiny-text'
import ChatTextarea from './ChatTextarea'
import MessageRegular from './Message/Regular'
import { ToolCallContent } from './Message/ToolCallContent'
import ToolCallTag from './Message/ToolCallTag'
import SessionSelector from './SessionSelector'
import ChatSpinner from './Spinner'
import ToolcallProgressUpdate from './ToolcallProgressUpdate'
import ShareTemplateDialog from './ShareTemplateDialog'

import { useConfigs } from '@/contexts/configs'
import 'react-photo-view/dist/react-photo-view.css'
import { DEFAULT_SYSTEM_PROMPT } from '@/constants'
import { ModelInfo, ToolInfo } from '@/api/model'
import { Button } from '@/components/ui/button'
import { Share2 } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'
import { useQueryClient } from '@tanstack/react-query'
import MixedContent, { MixedContentImages, MixedContentText } from './Message/MixedContent'


type ChatInterfaceProps = {
  canvasId: string
  sessionList: Session[]
  setSessionList: Dispatch<SetStateAction<Session[]>>
  sessionId: string
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  canvasId,
  sessionList,
  setSessionList,
  sessionId: searchSessionId,
}) => {
  const { t } = useTranslation()
  const [session, setSession] = useState<Session | null>(null)
  const { initCanvas, setInitCanvas } = useConfigs()
  const { authStatus } = useAuth()
  const [showShareDialog, setShowShareDialog] = useState(false)
  const queryClient = useQueryClient()

  useEffect(() => {
    if (sessionList.length > 0) {
      let _session = null
      if (searchSessionId) {
        _session = sessionList.find((s) => s.id === searchSessionId) || null
      } else {
        _session = sessionList[0]
      }
      setSession(_session)
    } else {
      setSession(null)
    }
  }, [sessionList, searchSessionId])

  type MessageWithUid = Message & { __uid?: string; __isError?: boolean; __isInfo?: boolean }
  const [messages, setMessages] = useState<MessageWithUid[]>([])
  const [pending, setPending] = useState<PendingType>(
    initCanvas ? 'text' : false
  )
  const mergedToolCallIds = useRef<string[]>([])
  // Number of messages in prev at the moment the user hits "Send".
  // all_messages events whose length <= this value only echo (possibly truncated)
  // history — no genuinely new agent responses yet. We skip those updates to
  // avoid React key churn and insertBefore DOM crashes caused by context-window
  // truncation making the server-side state shorter than the frontend's prev.
  const preSendMessageCount = useRef(0)

  const ensureMessageUid = (message: MessageWithUid): MessageWithUid => {
    if (!message.__uid) {
      message.__uid = nanoid()
    }
    return message
  }

  const ensureMessagesUids = (messages: MessageWithUid[]): MessageWithUid[] =>
    messages.map((message) => ensureMessageUid(message))

  const sessionId = session?.id ?? searchSessionId

  const sessionIdRef = useRef<string>(session?.id || nanoid())
  // Tracks whether a 'done' WebSocket event arrived while initChat was still
  // awaiting its fetches.  If true, initChat must not overwrite the correct
  // pending=false with a stale pending='tool'.
  const doneReceivedDuringInitRef = useRef(false)
  const [expandingToolCalls, setExpandingToolCalls] = useState<string[]>([])
  const [pendingToolConfirmations, setPendingToolConfirmations] = useState<
    string[]
  >([])
  // Ref mirror of pendingToolConfirmations so handleToolCallArguments can read
  // the latest value without being listed as a useCallback dependency (which
  // would cause the event handler to be re-registered on every confirmation
  // toggle → double event subscription → duplicate argument appends).
  const pendingToolConfirmationsRef = useRef<string[]>([])
  useEffect(() => {
    pendingToolConfirmationsRef.current = pendingToolConfirmations
  }, [pendingToolConfirmations])

  type PendingBatchApproval = {
    batch_id: string
    tool_calls: Array<{ id: string; name: string; arguments: Record<string, any> }>
  }
  const [pendingBatchApprovals, setPendingBatchApprovals] = useState<PendingBatchApproval[]>([])

  const scrollRef = useRef<HTMLDivElement>(null)
  const isAtBottomRef = useRef(false)

  const scrollToBottom = useCallback(() => {
    if (!isAtBottomRef.current) {
      return
    }
    setTimeout(() => {
      scrollRef.current?.scrollTo({
        top: scrollRef.current!.scrollHeight,
        behavior: 'smooth',
      })
    }, 200)
  }, [])

  const mergeToolCallResult = (messages: MessageWithUid[]) => {
    const messagesWithToolCallResult = messages.map((message, index) => {
      if (message.role === 'assistant' && message.tool_calls) {
        for (const toolCall of message.tool_calls) {
          // From the next message, find the tool call result
          for (let i = index + 1; i < messages.length; i++) {
            const nextMessage = messages[i]
            if (
              nextMessage.role === 'tool' &&
              nextMessage.tool_call_id === toolCall.id
            ) {
              toolCall.result = nextMessage.content
              mergedToolCallIds.current.push(toolCall.id)
            }
          }
        }
      }
      return ensureMessageUid(message)
    })

    return messagesWithToolCallResult
  }

  const handleDelta = useCallback(
    (data: TEvents['Socket::Session::Delta']) => {
      if (data.session_id && data.session_id !== sessionId) {
        return
      }

      setPending('text')
      setMessages(
        produce((prev) => {
          const last = prev.at(-1)
          if (
            last?.role === 'assistant' &&
            last.content != null &&
            last.tool_calls == null
          ) {
            if (typeof last.content === 'string') {
              last.content += data.text
            } else if (
              last.content &&
              last.content.at(-1) &&
              last.content.at(-1)!.type === 'text'
            ) {
              ; (last.content.at(-1) as { text: string }).text += data.text
            }
          } else {
            prev.push(
              ensureMessageUid({
                role: 'assistant',
                content: data.text,
              })
            )
          }
        })
      )
      scrollToBottom()
    },
    [sessionId, scrollToBottom]
  )

  const handleToolCall = useCallback(
    (data: TEvents['Socket::Session::ToolCall']) => {
      if (data.session_id && data.session_id !== sessionId) {
        return
      }

      // Use functional update so the duplicate check reads the latest state,
      // not a stale closure value (messages was missing from deps before).
      // Also move setPending outside of produce — calling setState inside a
      // produce callback triggers a nested React update during state computation,
      // which can corrupt the fiber tree and cause insertBefore DOM crashes.
      setMessages((prev) => {
        const exists = prev.some(
          (m) =>
            m.role === 'assistant' &&
            m.tool_calls?.some((t) => t.id === data.id)
        )
        if (exists) return prev
        console.log('👇tool_call event get', data)
        return produce(prev, (draft) => {
          draft.push(
            ensureMessageUid({
              role: 'assistant',
              content: '',
              tool_calls: [
                {
                  type: 'function',
                  function: { name: data.name, arguments: '' },
                  id: data.id,
                },
              ],
            })
          )
        })
      })
      setPending('tool')

      setExpandingToolCalls(
        produce((prev) => {
          prev.push(data.id)
        })
      )
    },
    [sessionId]
  )

  const handleToolCallPendingConfirmation = useCallback(
    (data: TEvents['Socket::Session::ToolCallPendingConfirmation']) => {
      if (data.session_id && data.session_id !== sessionId) {
        return
      }

      // Use functional update so the duplicate check reads the latest state,
      // not a stale closure value — same pattern as handleToolCall.
      // Stale-closure duplicate check lets the same tool_call_id get pushed
      // twice before React re-renders, causing key collisions → insertBefore crash.
      setMessages((prev) => {
        const exists = prev.some(
          (m) =>
            m.role === 'assistant' &&
            m.tool_calls?.some((t) => t.id === data.id)
        )
        if (exists) return prev
        console.log('👇tool_call_pending_confirmation event get', data)
        return produce(prev, (draft) => {
          draft.push(
            ensureMessageUid({
              role: 'assistant',
              content: '',
              tool_calls: [
                {
                  type: 'function',
                  function: {
                    name: data.name,
                    arguments: data.arguments,
                  },
                  id: data.id,
                },
              ],
            })
          )
        })
      })
      setPending('tool')

      setPendingToolConfirmations(
        produce((prev) => {
          prev.push(data.id)
        })
      )

      // 自动展开需要确认的工具调用
      setExpandingToolCalls(
        produce((prev) => {
          if (!prev.includes(data.id)) {
            prev.push(data.id)
          }
        })
      )
    },
    [sessionId]
  )

  const handleToolCallConfirmed = useCallback(
    (data: TEvents['Socket::Session::ToolCallConfirmed']) => {
      if (data.session_id && data.session_id !== sessionId) {
        return
      }

      setPendingToolConfirmations(
        produce((prev) => {
          return prev.filter((id) => id !== data.id)
        })
      )

      setExpandingToolCalls(
        produce((prev) => {
          if (!prev.includes(data.id)) {
            prev.push(data.id)
          }
        })
      )
    },
    [sessionId]
  )

  const handleToolCallCancelled = useCallback(
    (data: TEvents['Socket::Session::ToolCallCancelled']) => {
      if (data.session_id && data.session_id !== sessionId) {
        return
      }

      setPendingToolConfirmations(
        produce((prev) => {
          return prev.filter((id) => id !== data.id)
        })
      )

      // 更新工具调用的状态
      setMessages(
        produce((prev) => {
          prev.forEach((msg) => {
            if (msg.role === 'assistant' && msg.tool_calls) {
              msg.tool_calls.forEach((tc) => {
                if (tc.id === data.id) {
                  // 添加取消状态标记
                  tc.result = '工具调用已取消'
                }
              })
            }
          })
        })
      )
    },
    [sessionId]
  )

  const handleToolApprovalRequest = useCallback(
    (data: TEvents['Socket::Session::ToolApprovalRequest']) => {
      if (data.session_id && data.session_id !== sessionId) return
      setPendingBatchApprovals((prev) => [
        ...prev,
        { batch_id: data.batch_id, tool_calls: data.tool_calls },
      ])
      // Auto-expand all tool calls in the batch so the user sees them
      setExpandingToolCalls((prev) => {
        const newIds = data.tool_calls
          .map((tc) => tc.id)
          .filter((id) => !prev.includes(id))
        return [...prev, ...newIds]
      })
    },
    [sessionId]
  )

  const handleToolBatchApproved = useCallback(
    (data: TEvents['Socket::Session::ToolBatchApproved']) => {
      if (data.session_id && data.session_id !== sessionId) return
      setPendingBatchApprovals((prev) =>
        prev.filter((b) => b.batch_id !== data.batch_id)
      )
    },
    [sessionId]
  )

  const handleToolBatchRejected = useCallback(
    (data: TEvents['Socket::Session::ToolBatchRejected']) => {
      if (data.session_id && data.session_id !== sessionId) return
      setPendingBatchApprovals((prev) => {
        const batch = prev.find((b) => b.batch_id === data.batch_id)
        if (batch) {
          // Mark the rejected tool calls as cancelled in messages
          setMessages(
            produce((msgs) => {
              msgs.forEach((msg) => {
                if (msg.role === 'assistant' && msg.tool_calls) {
                  msg.tool_calls.forEach((tc) => {
                    if (batch.tool_calls.some((btc) => btc.id === tc.id)) {
                      tc.result = '工具调用已取消'
                    }
                  })
                }
              })
            })
          )
        }
        return prev.filter((b) => b.batch_id !== data.batch_id)
      })
    },
    [sessionId]
  )

  const handleToolCallArguments = useCallback(
    (data: TEvents['Socket::Session::ToolCallArguments']) => {
      if (data.session_id && data.session_id !== sessionId) {
        return
      }

      // Move setPending outside produce — setState inside produce causes a
      // nested React update during state computation → insertBefore DOM crash.
      setPending('tool')
      setMessages(
        produce((prev) => {
          const lastMessage = prev.find(
            (m) =>
              m.role === 'assistant' &&
              m.tool_calls &&
              m.tool_calls.find((t) => t.id == data.id)
          ) as AssistantMessage

          if (lastMessage) {
            const toolCall = lastMessage.tool_calls!.find(
              (t) => t.id == data.id
            )
            if (toolCall) {
              // 检查是否是待确认的工具调用，如果是则跳过参数追加
              if (pendingToolConfirmationsRef.current.includes(data.id)) {
                return
              }
              toolCall.function.arguments += data.text
            }
          }
        })
      )
      scrollToBottom()
    },
    [sessionId, scrollToBottom]
  )

  const handleToolCallResult = useCallback(
    (data: TEvents['Socket::Session::ToolCallResult']) => {
      console.log('😘🖼️tool_call_result event get', data)
      if (data.session_id && data.session_id !== sessionId) {
        return
      }
      // TODO: support other non string types of returning content like image_url
      if (data.message.content) {
        setMessages(
          produce((prev) => {
            prev.forEach((m) => {
              if (m.role === 'assistant' && m.tool_calls) {
                m.tool_calls.forEach((t) => {
                  if (t.id === data.id) {
                    t.result = data.message.content
                  }
                })
              }
            })
          })
        )
        // Mark this tool result as merged so it is hidden (shown inside the green box)
        // even if handleAllMessages fires before mergeToolCallResult has a chance to run.
        if (!mergedToolCallIds.current.includes(data.id)) {
          mergedToolCallIds.current.push(data.id)
        }
      }
    },
    [canvasId, sessionId]
  )

  const handleImageGenerated = useCallback(
    (data: TEvents['Socket::Session::ImageGenerated']) => {
      if (
        data.canvas_id &&
        data.canvas_id !== canvasId &&
        data.session_id !== sessionId
      ) {
        return
      }

      console.log('⭐️dispatching image_generated', data)
      setPending('image')
    },
    [canvasId, sessionId]
  )

  const handleAllMessages = useCallback(
    (data: TEvents['Socket::Session::AllMessages']) => {
      if (data.session_id && data.session_id !== sessionId) {
        return
      }

      // Use functional update so we can read the current messages and
      // reuse their __uid values.  Generating fresh UIDs on every all_messages
      // event causes React key churn that triggers insertBefore DOM crashes.
      //
      // We match by tool_call_id first (stable across concurrent events such as
      // handleToolCall appending a message just before all_messages arrives),
      // then fall back to positional index for non-tool-call messages.
      //
      // Guard: when the backend's context-window truncation reduces the message
      // count, all_messages contains fewer messages than the frontend's prev.
      // Replacing prev with this shorter list drops historical messages from the
      // UI and causes key churn → insertBefore DOM crash.  Skip any all_messages
      // event whose message count doesn't exceed the pre-send baseline; the
      // individual streaming events (handleDelta / handleToolCall /
      // handleToolCallResult) already keep the UI up-to-date for those rounds.
      // mergeToolCallResult preserves length so data.messages.length is sufficient.
      if (data.messages.length <= preSendMessageCount.current) {
        scrollToBottom()
        return
      }
      setMessages((prev) => {
        // Build stable lookup: first tool_call id of each assistant message → __uid
        // AND collect tool_calls per assistant message by positional index
        const uidByToolCallId = new Map<string, string>()
        const prevAssistantToolCalls: Array<{ idx: number; toolCalls: any[] }>[] = []
        for (let i = 0; i < prev.length; i++) {
          const m = prev[i]
          if (m.role === 'assistant' && (m as any).tool_calls?.length && (m as MessageWithUid).__uid) {
            const firstId = (m as any).tool_calls[0].id as string
            uidByToolCallId.set(firstId, (m as MessageWithUid).__uid!)
          }
          if (m.role === 'assistant' && (m as any).tool_calls?.length) {
            prevAssistantToolCalls.push({ idx: i, toolCalls: (m as any).tool_calls })
          }
        }

        const merged = mergeToolCallResult(data.messages)

        // Build mapping: which server assistant messages already have tool_calls
        const mergedAssistantIndices: number[] = []
        for (let i = 0; i < merged.length; i++) {
          if (merged[i].role === 'assistant') {
            mergedAssistantIndices.push(i)
          }
        }

        return merged.map((msg, idx) => {
          // Prefer match by tool_call_id — unaffected by concurrent appends
          let existingUid: string | undefined
          if (msg.role === 'assistant' && (msg as any).tool_calls?.length) {
            existingUid = uidByToolCallId.get((msg as any).tool_calls[0].id)
          }
          // Fall back to positional match for regular messages
          existingUid = existingUid ?? (prev[idx] as MessageWithUid)?.__uid ?? (msg as MessageWithUid).__uid

          if (msg.role === 'assistant') {
            // Find the matching prev message by __uid (reliable) to preserve/restore tool_calls
            const prevMsg = existingUid
              ? (prev.find(m => (m as MessageWithUid).__uid === existingUid) as any)
              : null

            if (!(msg as any).tool_calls?.length) {
              // LangGraph stripped tool_calls after resolution — restore from prev using __uid match.
              // Falls back to positional match only if __uid lookup fails (e.g. brand-new message).
              const hasContent = typeof msg.content === 'string' && msg.content.trim().length > 0
              if (!hasContent) {
                const prevToolCalls = prevMsg?.tool_calls?.length
                  ? prevMsg.tool_calls
                  : (() => {
                      // positional fallback (legacy — kept for non-uid messages)
                      const assistantPos = mergedAssistantIndices.indexOf(idx)
                      return (assistantPos >= 0 && assistantPos < prevAssistantToolCalls.length)
                        ? prevAssistantToolCalls[assistantPos]?.toolCalls
                        : undefined
                    })()
                if (prevToolCalls?.length) {
                  ; (msg as any).tool_calls = prevToolCalls
                }
              }
            } else if (prevMsg?.tool_calls?.length) {
              // tool_calls present in server data but mergeToolCallResult may not have set
              // toolCall.result yet (tool ran but values chunk arrived before tool_call_result event,
              // or LangGraph didn't include the tool result message in this values chunk).
              // Preserve any results already set in prev so the green box keeps showing them.
              ;(msg as any).tool_calls.forEach((tc: any) => {
                if (!tc.result) {
                  const prevTc = prevMsg.tool_calls.find((pt: any) => pt.id === tc.id)
                  if (prevTc?.result) tc.result = prevTc.result
                }
              })
            }
          }

          return { ...msg, __uid: existingUid }
        })
      })
      scrollToBottom()
    },
    [sessionId, scrollToBottom]
  )

  const handleDone = useCallback(
    (data: TEvents['Socket::Session::Done']) => {
      if (data.session_id && data.session_id !== sessionId) {
        return
      }

      // Signal initChat not to overwrite this with a stale 'tool' pending value
      doneReceivedDuringInitRef.current = true
      setPending(false)
      scrollToBottom()

      // 聊天输出完毕后更新余额
      if (authStatus.is_logged_in) {
        queryClient.invalidateQueries({ queryKey: ['balance'] })
      }
    },
    [sessionId, scrollToBottom, authStatus.is_logged_in, queryClient]
  )

  const handleError = useCallback(
    (data: TEvents['Socket::Session::Error']) => {
      if (data.session_id && data.session_id !== sessionId) return
      setPending(false)
      // Display the error as a message in the session instead of a global toast,
      // so the user sees it in context and can retry inline.
      setMessages((prev) => [
        ...prev,
        ensureMessageUid({
          role: 'assistant',
          content: data.error ?? 'Unknown error',
          __isError: true,
        } as MessageWithUid),
      ])
    },
    [sessionId]
  )

  const handleInfo = useCallback((data: TEvents['Socket::Session::Info']) => {
    if (data.session_id && data.session_id !== sessionId) return
    // Display info messages in-session instead of global toast
    setMessages((prev) => [
      ...prev,
      ensureMessageUid({
        role: 'assistant',
        content: data.info,
        __isInfo: true,
      } as MessageWithUid),
    ])
  }, [sessionId])

  // Scroll listener — only needs to run once (ref is stable)
  useEffect(() => {
    const handleScroll = () => {
      if (scrollRef.current) {
        isAtBottomRef.current =
          scrollRef.current.scrollHeight - scrollRef.current.scrollTop <=
          scrollRef.current.clientHeight + 1
      }
    }
    const scrollEl = scrollRef.current
    scrollEl?.addEventListener('scroll', handleScroll)
    return () => {
      scrollEl?.removeEventListener('scroll', handleScroll)
    }
  }, [])

  // Event bus subscriptions — re-subscribe whenever any handler changes
  // (handlers change when sessionId changes via useCallback deps)
  useEffect(() => {
    eventBus.on('Socket::Session::Delta', handleDelta)
    eventBus.on('Socket::Session::ToolCall', handleToolCall)
    eventBus.on(
      'Socket::Session::ToolCallPendingConfirmation',
      handleToolCallPendingConfirmation
    )
    eventBus.on('Socket::Session::ToolCallConfirmed', handleToolCallConfirmed)
    eventBus.on('Socket::Session::ToolCallCancelled', handleToolCallCancelled)
    eventBus.on('Socket::Session::ToolCallArguments', handleToolCallArguments)
    eventBus.on('Socket::Session::ToolCallResult', handleToolCallResult)
    eventBus.on('Socket::Session::ImageGenerated', handleImageGenerated)
    eventBus.on('Socket::Session::AllMessages', handleAllMessages)
    eventBus.on('Socket::Session::Done', handleDone)
    eventBus.on('Socket::Session::Error', handleError)
    eventBus.on('Socket::Session::Info', handleInfo)
    eventBus.on('Socket::Session::ToolApprovalRequest', handleToolApprovalRequest)
    eventBus.on('Socket::Session::ToolBatchApproved', handleToolBatchApproved)
    eventBus.on('Socket::Session::ToolBatchRejected', handleToolBatchRejected)
    return () => {
      eventBus.off('Socket::Session::Delta', handleDelta)
      eventBus.off('Socket::Session::ToolCall', handleToolCall)
      eventBus.off(
        'Socket::Session::ToolCallPendingConfirmation',
        handleToolCallPendingConfirmation
      )
      eventBus.off(
        'Socket::Session::ToolCallConfirmed',
        handleToolCallConfirmed
      )
      eventBus.off(
        'Socket::Session::ToolCallCancelled',
        handleToolCallCancelled
      )
      eventBus.off(
        'Socket::Session::ToolCallArguments',
        handleToolCallArguments
      )
      eventBus.off('Socket::Session::ToolCallResult', handleToolCallResult)
      eventBus.off('Socket::Session::ImageGenerated', handleImageGenerated)
      eventBus.off('Socket::Session::AllMessages', handleAllMessages)
      eventBus.off('Socket::Session::Done', handleDone)
      eventBus.off('Socket::Session::Error', handleError)
      eventBus.off('Socket::Session::Info', handleInfo)
      eventBus.off('Socket::Session::ToolApprovalRequest', handleToolApprovalRequest)
      eventBus.off('Socket::Session::ToolBatchApproved', handleToolBatchApproved)
      eventBus.off('Socket::Session::ToolBatchRejected', handleToolBatchRejected)
    }
  }, [
    handleDelta,
    handleToolCall,
    handleToolCallPendingConfirmation,
    handleToolCallConfirmed,
    handleToolCallCancelled,
    handleToolCallArguments,
    handleToolCallResult,
    handleImageGenerated,
    handleAllMessages,
    handleDone,
    handleError,
    handleInfo,
    handleToolApprovalRequest,
    handleToolBatchApproved,
    handleToolBatchRejected,
  ])

  const initChat = useCallback(async () => {
    if (!sessionId) {
      return
    }

    sessionIdRef.current = sessionId
    // Reset so we can detect a 'done' event that fires during our fetches
    doneReceivedDuringInitRef.current = false

    const [msgResp, runningResp] = await Promise.all([
      fetch('/api/chat_session/' + sessionId),
      fetch('/api/session/' + sessionId + '/running'),
    ])

    const data = msgResp.ok ? await msgResp.json() : []
    const running: boolean = runningResp.ok
      ? ((await runningResp.json())?.running ?? false)
      : false
    const msgs = data?.length ? data : []

    setMessages(mergeToolCallResult(msgs))
    if (msgs.length > 0) {
      setInitCanvas(false)
    }

    // If a 'done' WebSocket event arrived while we were awaiting the fetches,
    // handleDone already set pending=false — skip to avoid overwriting it with
    // a stale 'tool' value and leaving the spinner stuck forever.
    if (!doneReceivedDuringInitRef.current) {
      setPending(running ? 'tool' : false)
    }

    scrollToBottom()
  }, [sessionId, scrollToBottom, setInitCanvas])

  useEffect(() => {
    initChat()
  }, [sessionId, initChat])

  const onSelectSession = (sessionId: string) => {
    setSession(sessionList.find((s) => s.id === sessionId) || null)
    window.history.pushState(
      {},
      '',
      `/canvas/${canvasId}?sessionId=${sessionId}`
    )
  }

  const onClickNewChat = () => {
    const newSession: Session = {
      id: nanoid(),
      title: t('chat:newChat'),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      model: session?.model || 'gpt-4o',
      provider: session?.provider || 'openai',
    }

    setSessionList((prev) => [...prev, newSession])
    onSelectSession(newSession.id)
  }

  const onSendMessages = useCallback(
    (data: Message[], configs: { toolList: ToolInfo[] }) => {
      preSendMessageCount.current = data.length
      setPending('text')
      setMessages(ensureMessagesUids(data))

      sendMessages({
        sessionId: sessionId!,
        canvasId: canvasId,
        newMessages: data,
        textModel: undefined,   // 向后兼容：text model 现在通过 toolList 中 type='text' 的工具传递
        toolList: configs.toolList,
        systemPrompt:
          localStorage.getItem('system_prompt') || DEFAULT_SYSTEM_PROMPT,
      })

      if (searchSessionId !== sessionId) {
        window.history.pushState(
          {},
          '',
          `/canvas/${canvasId}?sessionId=${sessionId}`
        )
      }

      scrollToBottom()
    },
    [canvasId, sessionId, searchSessionId, scrollToBottom]
  )

  const handleCancelChat = useCallback(() => {
    setPending(false)
  }, [])

  return (
    <PhotoProvider>
      <div className='flex flex-col h-screen relative'>
        {/* Chat messages */}

        <header className='flex items-center px-2 py-2 absolute top-0 z-1 w-full'>
          <div className='flex-1 min-w-0'>
            <SessionSelector
              session={session}
              sessionList={sessionList}
              onClickNewChat={onClickNewChat}
              onSelectSession={onSelectSession}
            />
          </div>

          {/* Share Template Button */}
          {/* {authStatus.is_logged_in && (
            <Button
              variant="outline"
              size="sm"
              className="ml-2 shrink-0"
              onClick={() => setShowShareDialog(true)}
            >
              <Share2 className="h-4 w-4 mr-1" />
            </Button>
          )} */}

          <Blur className='absolute top-0 left-0 right-0 h-full -z-1' />
        </header>

        <ScrollArea className='h-[calc(100vh-45px)]' viewportRef={scrollRef}>
          {messages.length > 0 ? (
            <div className='flex flex-col flex-1 px-4 pb-50 pt-15'>
              {/* Messages */}
              {messages.map((message, idx) => (
                <div key={message.__uid ?? `message-${idx}`} className='flex flex-col gap-4 mb-2'>
                  {/* Error message — displayed in-session instead of global toast */}
                  {(message as MessageWithUid).__isError && typeof message.content === 'string' && (
                    <div className='flex items-start gap-2 text-destructive text-sm py-2'>
                      <span className='shrink-0 mt-0.5'>⚠️</span>
                      <span>{message.content}</span>
                    </div>
                  )}
                  {/* Info message — displayed in-session instead of global toast */}
                  {(message as MessageWithUid).__isInfo && typeof message.content === 'string' && (
                    <div className='flex items-start gap-2 text-muted-foreground text-sm py-2'>
                      <span className='shrink-0 mt-0.5'>ℹ️</span>
                      <span>{message.content}</span>
                    </div>
                  )}
                  {/* Regular message content */}
                  {!(message as MessageWithUid).__isError && !(message as MessageWithUid).__isInfo && typeof message.content == 'string' &&
                    (message.role !== 'tool' ? (
                      <MessageRegular
                        message={message}
                        content={message.content}
                      />
                    ) : message.tool_call_id &&
                      mergedToolCallIds.current.includes(
                        message.tool_call_id
                      ) ? (
                      <></>
                    ) : (
                      <ToolCallContent
                        expandingToolCalls={expandingToolCalls}
                        message={message}
                      />
                    ))}

                  {/* 混合内容消息的文本部分 - 显示在聊天框内 */}
                  {Array.isArray(message.content) && (
                    <>
                      <MixedContentImages
                        contents={message.content}
                      />
                      <MixedContentText
                        message={message}
                        contents={message.content}
                      />
                    </>
                  )}

                  {message.role === 'assistant' &&
                    message.tool_calls &&
                    message.tool_calls.at(-1)?.function.name != 'finish' &&
                    message.tool_calls.map((toolCall, i) => {
                      // Check if this tool call belongs to a pending batch approval
                      const pendingBatch = pendingBatchApprovals.find((b) =>
                        b.tool_calls.some((btc) => btc.id === toolCall.id)
                      )
                      // Also check legacy per-call confirmations
                      const needsLegacyConfirmation = pendingToolConfirmations.includes(toolCall.id)

                      return (
                        <ToolCallTag
                          key={toolCall.id}
                          toolCall={toolCall}
                          isExpanded={expandingToolCalls.includes(toolCall.id)}
                          onToggleExpand={() => {
                            if (expandingToolCalls.includes(toolCall.id)) {
                              setExpandingToolCalls((prev) =>
                                prev.filter((id) => id !== toolCall.id)
                              )
                            } else {
                              setExpandingToolCalls((prev) => [
                                ...prev,
                                toolCall.id,
                              ])
                            }
                          }}
                          requiresConfirmation={!!pendingBatch || needsLegacyConfirmation}
                          onConfirm={() => {
                            if (pendingBatch) {
                              // Batch approval: approve the whole batch
                              fetch('/api/batch_tool_approval', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                  session_id: sessionId,
                                  batch_id: pendingBatch.batch_id,
                                  approved: true,
                                  tool_calls: pendingBatch.tool_calls,
                                }),
                              })
                            } else {
                              // Legacy per-call confirmation
                              fetch('/api/tool_confirmation', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                  session_id: sessionId,
                                  tool_call_id: toolCall.id,
                                  confirmed: true,
                                }),
                              })
                            }
                          }}
                          onCancel={() => {
                            if (pendingBatch) {
                              // Batch rejection: reject the whole batch
                              fetch('/api/batch_tool_approval', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                  session_id: sessionId,
                                  batch_id: pendingBatch.batch_id,
                                  approved: false,
                                  tool_calls: [],
                                }),
                              })
                            } else {
                              // Legacy per-call cancellation
                              fetch('/api/tool_confirmation', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                  session_id: sessionId,
                                  tool_call_id: toolCall.id,
                                  confirmed: false,
                                }),
                              })
                            }
                          }}
                        />
                      )
                    })}
                </div>
              ))}
              {/* Stable wrapper: keeps DOM structure fixed so React never needs
                  insertBefore on a node that may have been removed concurrently */}
              <div>
                {pending && <ChatSpinner pending={pending} />}
                {pending && sessionId && (
                  <ToolcallProgressUpdate sessionId={sessionId} />
                )}
              </div>
            </div>
          ) : (
            <motion.div className='flex flex-col h-full p-4 items-start justify-start pt-16 select-none'>
              <motion.span
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className='text-muted-foreground text-3xl'
              >
                <ShinyText text='Hello, Jaaz!' />
              </motion.span>
              <motion.span
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className='text-muted-foreground text-2xl'
              >
                <ShinyText text='How can I help you today?' />
              </motion.span>
            </motion.div>
          )}
        </ScrollArea>

        <div className='p-2 gap-2 sticky bottom-0'>
          <ChatTextarea
            sessionId={sessionId!}
            pending={!!pending}
            messages={messages}
            onSendMessages={onSendMessages}
            onCancelChat={handleCancelChat}
          />

          {/* 魔法生成组件 */}
          <ChatMagicGenerator
            sessionId={sessionId || ''}
            canvasId={canvasId}
            messages={messages}
            setMessages={setMessages}
            setPending={setPending}
            scrollToBottom={scrollToBottom}
          />
        </div>
      </div>

      {/* Share Template Dialog */}
      <ShareTemplateDialog
        open={showShareDialog}
        onOpenChange={setShowShareDialog}
        canvasId={canvasId}
        sessionId={sessionId || ''}
        messages={messages}
      />
    </PhotoProvider>
  )
}

export default ChatInterface
