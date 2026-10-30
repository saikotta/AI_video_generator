export interface UserResponse {
  id: string
  email: string
  created_at: string
}

export interface SceneResponse {
  id: string
  scene_order: number
  scene_text: string
  visual_prompt: string | null
  duration_seconds: number | null
  status: string
}

export interface ProjectResponse {
  id: string
  topic: string
  desired_duration_seconds: number
  aspect_ratio: string
  status: string
  current_stage: string | null
  progress_percent: number
  error_message: string | null
  created_at: string
  updated_at: string
  completed_at: string | null
}

export interface ProjectDetailResponse extends ProjectResponse {
  scenes: SceneResponse[]
}

export interface GenerationStartResponse {
  project_id: string
  message: string
}

export interface JobStatusResponse {
  project_id: string
  status: string
  current_stage: string | null
  progress_percent: number
  error_message: string | null
}

export type ProjectStatus =
  | 'queued'
  | 'generating_script'
  | 'splitting_scenes'
  | 'generating_visuals'
  | 'generating_voice'
  | 'assembling'
  | 'done'
  | 'failed'
