# Asset Picker Popover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `DropdownMenu`-based material asset picker in ChatTextarea with a `Popover` floating panel that includes a search input and a 3-column thumbnail grid.

**Architecture:** All changes are confined to `ChatTextarea.tsx`. Two new state variables control the popover open state and search query. The existing `materialFiles` query and `addAssetToChat` callback are reused unchanged. Filtering is purely client-side.

**Tech Stack:** React, shadcn `Popover` (`@/components/ui/popover`), shadcn `Input` (`@/components/ui/input`), `lucide-react` `Search` icon, Tailwind CSS.

---

## File Map

| Action | File |
|---|---|
| Modify | `react/src/components/chat/ChatTextarea.tsx` |

---

### Task 1: Add imports and state

**Files:**
- Modify: `react/src/components/chat/ChatTextarea.tsx:1-51` (imports) and `~:79-110` (state section)

- [ ] **Step 1: Add Popover and Input imports**

In `ChatTextarea.tsx`, add after the existing `Tooltip` import block (after line 51):

```tsx
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Input } from '@/components/ui/input'
```

Also add `Search` to the lucide-react import list (line 18-30):

```tsx
import {
  ArrowUp,
  Loader2,
  PlusIcon,
  Search,
  Square,
  XIcon,
  RectangleVertical,
  ChevronDown,
  Hash,
  Video,
  Music,
  Library,
} from 'lucide-react'
```

- [ ] **Step 2: Add assetPickerOpen and assetSearchQuery state**

After the existing `showDurationSlider` state (around line 100), add:

```tsx
const [assetPickerOpen, setAssetPickerOpen] = useState(false)
const [assetSearchQuery, setAssetSearchQuery] = useState('')
```

- [ ] **Step 3: Verify the file compiles with no errors**

Run: `cd react && npm run build 2>&1 | tail -20`

Expected: build succeeds (no TypeScript errors related to the new imports/state).

- [ ] **Step 4: Commit**

```bash
git add react/src/components/chat/ChatTextarea.tsx
git commit -m "feat: add asset picker popover state and imports"
```

---

### Task 2: Replace DropdownMenu with Popover

**Files:**
- Modify: `react/src/components/chat/ChatTextarea.tsx:748-795` (the material picker block)

- [ ] **Step 1: Replace the entire material picker block**

Find and replace the block currently at lines 748-795 (from `{/* Material library asset picker */}` to closing `}`):

**Remove this:**
```tsx
          {/* Material library asset picker */}
          {materialFiles.length > 0 && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  <Library className="size-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-56 max-h-72 overflow-y-auto">
                {materialFiles
                  .filter(
                    (r) =>
                      r &&
                      r.disk_name &&
                      r.serve_url &&
                      r.status &&
                      r.status === 'Active'
                  )
                  .map((rec) => (
                    <DropdownMenuItem
                      key={rec.asset_id || rec.fid}
                      onClick={() => addAssetToChat(rec)}
                      className="flex items-center gap-2"
                    >
                      {rec.file_type === 'image' && rec.serve_url && (
                        <img
                          src={rec.serve_url}
                          alt={rec.name || 'Material'}
                          className="size-6 rounded object-cover shrink-0"
                          onError={(e) => {
                            ;(e.target as HTMLImageElement).style.display = 'none'
                          }}
                        />
                      )}
                      {rec.file_type === 'video' && (
                        <Video className="size-4 shrink-0 text-muted-foreground" />
                      )}
                      {rec.file_type === 'audio' && (
                        <Music className="size-4 shrink-0 text-muted-foreground" />
                      )}
                      <span className="truncate text-sm">
                        {rec.name || `Asset ${rec.asset_id}`}
                      </span>
                    </DropdownMenuItem>
                  ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
```

**Add this:**
```tsx
          {/* Material library asset picker */}
          <Popover
            open={assetPickerOpen}
            onOpenChange={(open) => {
              setAssetPickerOpen(open)
              if (!open) setAssetSearchQuery('')
            }}
          >
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm">
                <Library className="size-4" />
              </Button>
            </PopoverTrigger>
            <PopoverContent align="start" className="w-80 p-0">
              {/* Search input */}
              <div className="flex items-center gap-2 px-3 py-2 border-b">
                <Search className="size-4 shrink-0 text-muted-foreground" />
                <Input
                  value={assetSearchQuery}
                  onChange={(e) => setAssetSearchQuery(e.target.value)}
                  placeholder="搜索素材…"
                  className="h-7 border-0 p-0 text-sm shadow-none focus-visible:ring-0"
                />
              </div>
              {/* Grid */}
              <div className="max-h-80 overflow-y-auto p-2">
                {(() => {
                  const filtered = materialFiles
                    .filter(
                      (r) =>
                        r &&
                        r.disk_name &&
                        r.serve_url &&
                        r.status === 'Active' &&
                        (r.name ?? '').toLowerCase().includes(assetSearchQuery.toLowerCase())
                    )
                  if (filtered.length === 0) {
                    return (
                      <p className="py-6 text-center text-sm text-muted-foreground">
                        无匹配素材
                      </p>
                    )
                  }
                  return (
                    <div className="grid grid-cols-3 gap-1.5">
                      {filtered.map((rec) => (
                        <button
                          key={rec.asset_id || rec.fid}
                          onClick={() => {
                            addAssetToChat(rec)
                            setAssetPickerOpen(false)
                          }}
                          className="group flex flex-col items-center gap-1 rounded-md p-1 hover:bg-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        >
                          {rec.file_type === 'image' && rec.serve_url ? (
                            <img
                              src={rec.serve_url}
                              alt={rec.name || 'Material'}
                              className="h-16 w-full rounded object-cover"
                              onError={(e) => {
                                ;(e.target as HTMLImageElement).style.display = 'none'
                              }}
                            />
                          ) : rec.file_type === 'video' ? (
                            <div className="flex h-16 w-full items-center justify-center rounded bg-muted">
                              <Video className="size-6 text-muted-foreground" />
                            </div>
                          ) : (
                            <div className="flex h-16 w-full items-center justify-center rounded bg-muted">
                              <Music className="size-6 text-muted-foreground" />
                            </div>
                          )}
                          <span className="w-full truncate text-center text-xs text-muted-foreground group-hover:text-foreground">
                            {rec.name || `Asset ${rec.asset_id}`}
                          </span>
                        </button>
                      ))}
                    </div>
                  )
                })()}
              </div>
            </PopoverContent>
          </Popover>
```

- [ ] **Step 2: Verify build**

Run: `cd react && npm run build 2>&1 | tail -20`

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 3: Manual smoke test**

Start dev server (`npm run dev`), open the chat page, click the Library button:
- Popover opens with search box and grid
- Typing a name in search filters the grid in real time
- Clicking a material adds it to the chat and closes the popover
- Clicking outside closes the popover and clears the search query
- With no matching results, "无匹配素材" is shown

- [ ] **Step 4: Commit**

```bash
git add react/src/components/chat/ChatTextarea.tsx
git commit -m "feat: replace asset picker dropdown with searchable popover grid"
```
