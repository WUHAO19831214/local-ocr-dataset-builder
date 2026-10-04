<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { Clipboard, ExternalLink, Play } from 'lucide-vue-next'

const languages = [
  'ja-JP,zh-Hans',
  'zh-Hans',
  'ja-JP',
  'zh-Hant,ja-JP,zh-Hans',
]

const processModes = [
  { label: '普通教材 OCR', value: 'normal' },
  { label: '物理/数学公式优先', value: 'formula' },
  { label: '高精度公式复核（较慢）', value: 'formula_vl' },
]

const forceOcrOptions = [
  { label: '开启', value: true },
  { label: '关闭', value: false },
]

const form = ref({
  pdf_path: '/Users/wuhao/my-pdf-tool/my-pdf-tool/dajiaderiyufudaoshu12_p1_2.pdf',
  output_root: '/Users/wuhao/my-pdf-tool/datasets',
  output_name: 'test_builder_p1_2',
  ocr_lang: languages[0],
  process_mode: processModes[0].value,
  force_ocr: true,
  export_word: true,
})

const job = ref(null)
const logs = ref([])
const error = ref('')
const copied = ref(false)
const busy = computed(() => job.value?.status === 'running' || job.value?.status === 'queued')
const outputPath = computed(() => job.value?.output_path || expectedOutputPath.value)
const expectedOutputPath = computed(() => {
  const root = form.value.output_root.replace(/\/+$/, '')
  return root && form.value.output_name ? `${root}/${form.value.output_name}` : ''
})

watch(
  () => form.value.process_mode,
  (mode) => {
    form.value.force_ocr = mode === 'normal'
  }
)

let pollTimer = null

async function startJob() {
  error.value = ''
  copied.value = false
  logs.value = []
  job.value = null

  try {
    const response = await fetch('/api/jobs/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form.value),
    })

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}))
      throw new Error(payload.detail || `HTTP ${response.status}`)
    }

    const payload = await response.json()
    job.value = {
      job_id: payload.job_id,
      status: payload.status,
      progress_stage: 'queued',
      output_path: null,
      error: null,
    }
    await pollJob()
    pollTimer = window.setInterval(pollJob, 1200)
  } catch (err) {
    error.value = err.message || String(err)
  }
}

async function pollJob() {
  if (!job.value?.job_id) return
  const [jobResponse, logsResponse] = await Promise.all([
    fetch(`/api/jobs/${job.value.job_id}`),
    fetch(`/api/jobs/${job.value.job_id}/logs`),
  ])

  if (jobResponse.ok) {
    job.value = await jobResponse.json()
  }
  if (logsResponse.ok) {
    const payload = await logsResponse.json()
    logs.value = payload.logs || []
  }

  if (job.value && ['success', 'failed'].includes(job.value.status)) {
    window.clearInterval(pollTimer)
    pollTimer = null
    if (job.value.error) error.value = job.value.error
  }
}

async function openOutputPath() {
  error.value = ''
  try {
    const response = await fetch('/api/system/open-path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: outputPath.value }),
    })
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}))
      throw new Error(payload.detail || `HTTP ${response.status}`)
    }
  } catch (err) {
    error.value = err.message || String(err)
  }
}

async function copyOutputPath() {
  copied.value = false
  error.value = ''
  try {
    await navigator.clipboard.writeText(outputPath.value)
    copied.value = true
    window.setTimeout(() => {
      copied.value = false
    }, 1600)
  } catch (err) {
    error.value = err.message || String(err)
  }
}

onBeforeUnmount(() => {
  if (pollTimer) window.clearInterval(pollTimer)
})
</script>

<template>
  <main class="app-shell">
    <header class="title-bar">
      <div>
        <p class="eyebrow">macOS desktop OCR utility</p>
        <h1>Local OCR Dataset Builder</h1>
      </div>
      <div class="status-pill" :class="job?.status || 'idle'">
        {{ job?.status || 'idle' }}
      </div>
    </header>

    <section class="workspace">
      <form class="form-panel" @submit.prevent="startJob">
        <label>
          <span>PDF 文件路径</span>
          <input v-model.trim="form.pdf_path" type="text" autocomplete="off" placeholder="/absolute/path/book.pdf" />
        </label>

        <label>
          <span>输出根目录</span>
          <input v-model.trim="form.output_root" type="text" autocomplete="off" placeholder="/absolute/path/datasets" />
        </label>

        <div class="form-grid">
          <label>
            <span>输出名称</span>
            <input v-model.trim="form.output_name" type="text" autocomplete="off" placeholder="book_dataset" />
          </label>

          <label>
            <span>OCR 语言</span>
            <select v-model="form.ocr_lang">
              <option v-for="language in languages" :key="language" :value="language">
                {{ language }}
              </option>
            </select>
          </label>

          <label>
            <span>处理模式</span>
            <select v-model="form.process_mode">
              <option v-for="mode in processModes" :key="mode.value" :value="mode.value">
                {{ mode.label }}
              </option>
            </select>
          </label>

          <label>
            <span>强制 OCR</span>
            <select v-model="form.force_ocr">
              <option v-for="option in forceOcrOptions" :key="String(option.value)" :value="option.value">
                {{ option.label }}
              </option>
            </select>
          </label>

          <label>
            <span>同时输出 Word</span>
            <select v-model="form.export_word">
              <option :value="true">是：原版式 + 可编辑</option>
              <option :value="false">否</option>
            </select>
          </label>
        </div>

        <p v-if="form.process_mode === 'formula'" class="mode-hint">
          物理/数学公式优先模式使用 Docling 公式增强；在本机处理公式较多的 PDF 时可能较慢，单份 PDF 最长运行 10 分钟。
        </p>
        <p v-if="form.process_mode === 'formula_vl'" class="mode-hint">
          高精度复核先做一次基础 OCR，再用本机 PaddleOCR-VL 复核公式及疑似损坏的段内分式；未能识别的独立公式保留原图。可编辑 Word 使用 Word 原生公式，无需 MathType。
        </p>
        <p v-if="form.export_word" class="mode-hint">
          原版式 Word 每页是一张原 PDF 页面图片，版式与公式外观保真；可编辑 Word 使用 OCR 结果，识别错误仍需校对。
        </p>

        <button class="primary-action" type="submit" :disabled="busy">
          <Play :size="18" />
          <span>{{ busy ? 'OCR 运行中' : '开始 OCR' }}</span>
        </button>
      </form>

      <aside class="status-panel">
        <div class="metric">
          <span>当前状态</span>
          <strong>{{ job?.progress_stage || 'ready' }}</strong>
        </div>
        <div class="path-block">
          <span>输出目录</span>
          <code>{{ outputPath || '等待输入' }}</code>
        </div>
        <div class="action-row">
          <button type="button" @click="copyOutputPath" :disabled="!outputPath" title="复制工作台加载路径">
            <Clipboard :size="17" />
            <span>{{ copied ? '已复制' : '复制路径' }}</span>
          </button>
          <button type="button" @click="openOutputPath" :disabled="job?.status !== 'success'" title="处理完成后在 Finder 打开输出目录">
            <ExternalLink :size="17" />
            <span>打开目录</span>
          </button>
        </div>
        <p v-if="error" class="error-text">{{ error }}</p>
      </aside>
    </section>

    <section class="log-panel">
      <div class="log-header">
        <span>实时日志</span>
        <span>{{ logs.length }} lines</span>
      </div>
      <pre>{{ logs.length ? logs.join('\n') : '等待开始 OCR...' }}</pre>
    </section>
  </main>
</template>
