import { api } from './client'
import type {
  GenerationStartResponse,
  JobStatusResponse,
  ProjectDetailResponse,
  ProjectResponse,
} from './types'

export async function createProject(
  topic: string,
  desiredDurationSeconds: number,
  aspectRatio: string = '16:9',
): Promise<GenerationStartResponse> {
  const resp = await api.post<any>('/projects', {
    topic,
    desired_duration_seconds: desiredDurationSeconds,
    aspect_ratio: aspectRatio,
  })
  return {
    project_id: resp.project_id || resp.id,
    message: resp.message || 'Generation queued',
  }
}

export async function listProjects(skip = 0, limit = 100): Promise<ProjectResponse[]> {
  const data = await api.get<{ projects: ProjectResponse[]; total: number } | ProjectResponse[]>(
    `/projects?skip=${skip}&limit=${limit}&offset=${skip}`
  )
  if (Array.isArray(data)) {
    return data
  }
  if (data && Array.isArray(data.projects)) {
    return data.projects
  }
  return []
}

export async function getProject(id: string): Promise<ProjectDetailResponse> {
  try {
    const report = await api.get<any>(`/projects/${id}/report`)
    return {
      id: report.project_id || report.id || id,
      topic: report.topic || '',
      desired_duration_seconds: report.desired_duration_seconds || 15,
      aspect_ratio: report.aspect_ratio || '16:9',
      status: report.status || 'queued',
      current_stage: report.current_stage || null,
      progress_percent: report.progress_percent ?? 0,
      error_message: report.error_message || null,
      created_at: report.created_at || new Date().toISOString(),
      updated_at: report.updated_at || report.created_at || new Date().toISOString(),
      completed_at: report.completed_at || null,
      scenes: (report.scenes || []).map((sc: any, idx: number) => ({
        id: sc.id || String(idx + 1),
        scene_order: sc.scene_order ?? idx + 1,
        scene_text: sc.scene_text || '',
        visual_prompt: sc.visual_prompt || null,
        duration_seconds: sc.duration_seconds || null,
        status: sc.status || 'done',
      })),
    }
  } catch {
    const proj = await api.get<any>(`/projects/${id}`)
    return {
      id: proj.id || id,
      topic: proj.topic || '',
      desired_duration_seconds: proj.desired_duration_seconds || 15,
      aspect_ratio: proj.aspect_ratio || '16:9',
      status: proj.status || 'queued',
      current_stage: proj.current_stage || null,
      progress_percent: proj.progress_percent ?? 0,
      error_message: proj.error_message || null,
      created_at: proj.created_at || new Date().toISOString(),
      updated_at: proj.updated_at || proj.created_at || new Date().toISOString(),
      completed_at: proj.completed_at || null,
      scenes: [],
    }
  }
}

export async function getProjectStatus(id: string): Promise<JobStatusResponse> {
  const resp = await api.get<any>(`/projects/${id}`)
  return {
    project_id: resp.id || resp.project_id || id,
    status: resp.status,
    current_stage: resp.current_stage || null,
    progress_percent: resp.progress_percent ?? 0,
    error_message: resp.error_message || null,
  }
}

export async function downloadVideo(id: string): Promise<Blob> {
  return api.download(`/projects/${id}/video`)
}
