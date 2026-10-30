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
  return api.post<GenerationStartResponse>('/projects', {
    topic,
    desired_duration_seconds: desiredDurationSeconds,
    aspect_ratio: aspectRatio,
  })
}

export async function listProjects(skip = 0, limit = 20): Promise<ProjectResponse[]> {
  return api.get<ProjectResponse[]>(`/projects?skip=${skip}&limit=${limit}`)
}

export async function getProject(id: string): Promise<ProjectDetailResponse> {
  return api.get<ProjectDetailResponse>(`/projects/${id}`)
}

export async function getProjectStatus(id: string): Promise<JobStatusResponse> {
  return api.get<JobStatusResponse>(`/projects/${id}/status`)
}

export async function downloadVideo(id: string): Promise<Blob> {
  return api.download(`/projects/${id}/download`)
}
