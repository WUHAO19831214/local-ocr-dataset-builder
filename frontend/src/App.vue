<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import { Clipboard, ExternalLink, Play } from 'lucide-vue-next'

const languages = [
  'ja-JP,zh-Hans',
  'zh-Hans',
  'ja-JP',
  'zh-Hant,ja-JP,zh-Hans',
]

const form = ref({
  pdf_path: '/Users/wuhao/my-pdf-tool/my-pdf-tool/dajiaderiyufudaoshu12_p1_2.pdf',
  output_root: '/Users/wuhao/my-pdf-tool/datasets',
  output_name: 'test_builder_p1_2',
  ocr_lang: languages[0],
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

        <div class="two-column">
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
        </div>

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
          <button type="button" @click="openOutputPath" :disabled="!outputPath" title="在 Finder 打开输出目录">
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

