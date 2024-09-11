from dotenv import load_dotenv

from IPython.display import Image, display
import os
import autogen
from autogen.coding import LocalCommandLineCodeExecutor

load_dotenv()

# API Configuracion para OpenAI
# apiConfig = [{
#        "model": "gpt-4o-mini",
#        "max_tokens": 500,
#        "api_key": os.getenv('OPENAI_API_KEY')
#    }]

# Configuración de la API para Mistral - preferida por su costo y rendimiento.
api_config = [{
        "model": "mistral-large-latest",
        "max_tokens": 500,  # Límite reducido de tokens para respuestas más concisas
        "api_key": os.getenv('MISTRAL_API_KEY'),
        "api_type": "mistral"
    }]

# Configuración para el LLM, incluyendo lógica de reintentos y semilla de caché para consistencia
llm_config = {
    "config_list": api_config, 
    "cache_seed": 42,  # Semilla de caché para un comportamiento consistente
    "retry_on_rate_limit": True,  # Reintentar si se alcanza el límite de la API
    "retry_on_timeout": True,     # Reintentar si hay un tiempo de espera
    "max_retries": 5,             # Número máximo de reintentos
    "retry_delay": 0.3,           # Retraso más corto para reintentos más rápidos
}

# Crear el AssistantAgent llamado "Asistente"
asistente = autogen.AssistantAgent(
    name="Asistente",
    llm_config=llm_config,
    system_message="Sé conciso, informativo, y trata de reducir las preguntas de seguimiento."
)

# Crear el agente que actúa como Proxy del Usuario
proxy_usuario = autogen.UserProxyAgent(
    name="Usuario - Proxy",
    human_input_mode="NEVER",  # No se requiere retroalimentación del usuario. Opciones: "ALWAYS", "TERMINATE", "NEVER"
    max_consecutive_auto_reply=5,  # Reducido el número máximo de intercambios a 5
    is_termination_msg=lambda x: x.get("contenido", "").rstrip().endswith("TERMINATE") or "tarea completada" in x.get("contenido", ""),  # Termina cuando se completa la tarea
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir="contenido"),  # Donde se ejecuta el código generado
        "last_n_messages": 2,  # Solo considera los últimos 2 mensajes para reducir el contexto y uso de tokens
    },
)

task_request = "Genera la grafica YTD de precios y volumen del Bitcoin en USD y guardalo en un archivo con el nombre output.png"

# Entrada dinámica de solicitud del usuario para cualquier tarea o gráfico
# task_request = input("Por favor, ingrese la tarea que desea que el asistente complete: ")
# task_request += " y guarda el resultado en el archivo output.png"

# Inicia una conversación para solicitar cualquier tarea de forma dinámica
proxy_usuario.initiate_chat(
    recipient=asistente,
    message=task_request
)

# Intenta mostrar cualquier imagen o resultado generado, si aplica
output_file = "contenido/output.png"  # Puedes definir cualquier archivo de salida esperado
try:
    image = Image(filename=output_file)
    display(image)
except FileNotFoundError:
    print(f"No se encontró la salida en {output_file}. Es posible que la tarea solicitada no produzca un resultado visual.")

