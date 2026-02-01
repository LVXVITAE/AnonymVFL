{{/*
扩展 chart 名称
*/}}
{{- define "mobile-mpc.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
创建完全限定的应用名称
*/}}
{{- define "mobile-mpc.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
创建 chart 名称和版本，用于 chart 标签
*/}}
{{- define "mobile-mpc.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
公共标签
*/}}
{{- define "mobile-mpc.labels" -}}
helm.sh/chart: {{ include "mobile-mpc.chart" . }}
{{ include "mobile-mpc.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.global.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
选择器标签
*/}}
{{- define "mobile-mpc.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mobile-mpc.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Company 节点标签
*/}}
{{- define "mobile-mpc.company.labels" -}}
{{ include "mobile-mpc.labels" . }}
app.kubernetes.io/component: company
{{- end }}

{{/*
Partner 节点标签
*/}}
{{- define "mobile-mpc.partner.labels" -}}
{{ include "mobile-mpc.labels" . }}
app.kubernetes.io/component: partner
{{- end }}

{{/*
Coordinator 节点标签
*/}}
{{- define "mobile-mpc.coordinator.labels" -}}
{{ include "mobile-mpc.labels" . }}
app.kubernetes.io/component: coordinator
{{- end }}

{{/*
WebUI 标签
*/}}
{{- define "mobile-mpc.webui.labels" -}}
{{ include "mobile-mpc.labels" . }}
app.kubernetes.io/component: webui
{{- end }}

{{/*
创建 ServiceAccount 名称
*/}}
{{- define "mobile-mpc.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "mobile-mpc.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
镜像地址
*/}}
{{- define "mobile-mpc.image" -}}
{{- $registry := .Values.global.imageRegistry -}}
{{- $repository := .image.repository -}}
{{- $tag := .image.tag | default .Chart.AppVersion -}}
{{- printf "%s%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
公共环境变量
*/}}
{{- define "mobile-mpc.commonEnv" -}}
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
- name: HOST_IP
  valueFrom:
    fieldRef:
      fieldPath: status.hostIP
{{- end }}

{{/*
持久化卷声明
*/}}
{{- define "mobile-mpc.volumes" -}}
{{- if .persistence.enabled }}
- name: data
  persistentVolumeClaim:
    claimName: {{ .name }}-pvc
{{- else }}
- name: data
  emptyDir: {}
{{- end }}
- name: models
  emptyDir: {}
- name: logs
  emptyDir: {}
{{- end }}

