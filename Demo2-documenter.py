from dotenv import load_dotenv
import autogen
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Cargar variables de entorno
load_dotenv()


# API Configuracion para OpenAI
# apiConfig = [{
#        "model": "gpt-4o-mini",
#        "max_tokens": 500,
#        "api_key": os.getenv('OPENAI_API_KEY')
#    }]

# API Configuracion para Mistral -- Preferimos usar esta para las pruebas ya que no hay que pagar y es tan buena como la de OpenAI
api_config = [{
    "model": "mistral-large-latest",
    "max_tokens": 2000,
    "api_key": os.getenv('MISTRAL_API_KEY') or 'your_mistral_api_key_here',  # Reemplaza con la clave API
    "api_type": "mistral"
}]

# Configuración del LLM
llm_config = {
    "config_list": api_config,
    "cache_seed": 42,  # Consistencia en resultados
    "retry_on_rate_limit": True,  # Reintentar al alcanzar el límite de la API
    "retry_on_timeout": True,     # Reintentar en caso de tiempo de espera
    "max_retries": 5,             # Número máximo de reintentos
    "retry_delay": 0.3,           # Retraso entre reintentos
}

# Configuración de temperaturas para creatividad
llm_config_high = llm_config.copy()
llm_config_high["temperature"] = 0.9  # Alta creatividad

llm_config_low = llm_config.copy()
llm_config_low["temperature"] = 0.5  # Baja creatividad, más estructura

llm_config_normal = llm_config.copy()
llm_config_normal["temperature"] = 0.7  # Nivel normal de creatividad

# Agente Proxy del Usuario
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="Administrador humano del proyecto.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
    human_input_mode="TERMINATE"
)

# Agente de Project Manager (incluye plan de entrega)
pm = autogen.AssistantAgent(
    name="Project_Manager",
    system_message=("Eres el Project Manager responsable de gestionar las entregas del proyecto. Debes generar un plan de entrega "
                    "detallando los hitos, plazos y asignación de recursos necesarios para llevar el proyecto a buen puerto."),
    llm_config=llm_config_high,
)

# Agente de Product Manager (responsable del PRD)
product_manager = autogen.AssistantAgent(
    name="Product_Manager",
    system_message=("Eres un Product Manager con experiencia. Tu tarea es crear el Documento de Requisitos del Producto (PRD), "
                    "detallando todas las características principales del proyecto y sus objetivos."),
    llm_config=llm_config_high,
)

# Agente de Analista de Negocios
ba = autogen.AssistantAgent(
    name="Business_Analyst",
    system_message=("Eres un Analista de Negocios especializado en identificar y documentar casos de uso y requisitos funcionales. "
                    "Debes crear un documento independiente que detalle todos los casos de uso para el proyecto. Además, debes generar un diagrama de los casos de uso."),
    llm_config=llm_config_low,
)

# Agente de Arquitecto (incluye diagramas y C4)
architect = autogen.AssistantAgent(
    name="Architect",
    system_message=("Eres un Arquitecto de Software experto en diseño de sistemas escalables. Tu tarea es generar un documento con la arquitectura del sistema "
                    "utilizando diagramas C4 y Mermaid, explicando cada diagrama y detallando todos los casos de uso técnicos."),
    llm_config=llm_config_normal,
)

# Agente de Ingeniero de Software (Coder)
coder = autogen.AssistantAgent(
    name="Coder",
    system_message=("Eres un Ingeniero de Software Senior. Debes proporcionar un plan detallado de implementación, "
                    "focalizándote en las mejores prácticas de codificación y algoritmos eficientes."),
    llm_config=llm_config_normal,
)

# Agente de QA (incluye BDD Gherkin)
qa = autogen.AssistantAgent(
    name="QA",
    system_message=("Eres un experto en pruebas y calidad. Debes generar un plan de pruebas detallado que cubra pruebas funcionales, "
                    "de integración y de regresión para asegurar la calidad del producto. También debes generar una especificación en Gherkin para la implementación BDD."),
    llm_config=llm_config_low,
)

# Agente de Documentación
class DocumentationAgent(autogen.AssistantAgent):
    def __init__(self, name, llm_config):
        super().__init__(name=name, llm_config=llm_config)
        self.documents = {}

    def contribute(self, agent_name, document_title, content):
        # Crear documento separado por cada agente y aspecto
        self.documents[document_title] = content

    def generate_document(self, title):
        # Crear una carpeta específica para la documentación
        folder_name = "documentation"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Generar archivo de documentación para cada aspecto
        doc_path = os.path.join(folder_name, f"{title}.md")
        with open(doc_path, "w") as file:
            file.write(f"# {title}\n\n{self.documents.get(title, 'Sin contenido')}")
        print(f"Documento '{title}' creado en la carpeta 'documentation': {doc_path}")
        return doc_path

# Instanciar el agente de documentación
doc_agent = DocumentationAgent(
    name="Documentation_Agent",
    llm_config=llm_config_normal
)

# Definir el grupo de agentes
agents = [user_proxy, product_manager, ba, architect, coder, qa, pm, doc_agent]

# Configuración del Grupo de Chat
groupchat = autogen.GroupChat(agents=agents, messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Función para que los agentes contribuyan a la documentación
def agent_contribute(agent, doc_agent, document_title, groupchat):
    response = safe_agent_reply(agent, groupchat)
    if response:
        doc_agent.contribute(agent.name, document_title, response)
        print(f"{agent.name} contribuyó a '{document_title}'.")

# Función para manejo seguro de respuestas
def safe_agent_reply(agent, groupchat):
    try:
        response = agent.generate_reply(groupchat.messages)
        return response.get('content', '') if isinstance(response, dict) else response
    except Exception as e:
        print(f"Error generando respuesta de {agent.name}: {e}")
        return None

# Función asíncrona para generar la documentación final
async def generate_document_async(doc_agent, title):
    await asyncio.to_thread(doc_agent.generate_document, title)

# Función para iniciar la conversación solicitando información del proyecto
def solicitar_detalles_proyecto():
    print("Por favor, describa el tipo de proyecto que desea construir:")
    return input("Descripción del proyecto: ")

# Función para iniciar el flujo del proyecto
def start_project():
    descripcion_proyecto = solicitar_detalles_proyecto()

    # Mensaje inicial del Proxy del Usuario
    groupchat.messages.append({
        "sender": user_proxy.name,
        "content": f"Estamos iniciando un proyecto: {descripcion_proyecto}",
        "role": "user"
    })

    # Fase 1: Contribuciones iniciales de los agentes
    agent_contribute(product_manager, doc_agent, "PRD - Documento de Requisitos del Producto", groupchat)
    agent_contribute(ba, doc_agent, "Casos de Uso", groupchat)
    agent_contribute(architect, doc_agent, "Arquitectura del Sistema", groupchat)
    agent_contribute(coder, doc_agent, "Plan de Implementación", groupchat)
    agent_contribute(qa, doc_agent, "Plan de Pruebas", groupchat)
    agent_contribute(qa, doc_agent, "BDD Gherkin - Especificaciones", groupchat)  # Nueva contribución para Gherkin
    agent_contribute(pm, doc_agent, "Plan de Entrega", groupchat)

    # Generar los documentos de manera asíncrona
    asyncio.run(generate_document_async(doc_agent, "PRD - Documento de Requisitos del Producto"))
    asyncio.run(generate_document_async(doc_agent, "Casos de Uso"))
    asyncio.run(generate_document_async(doc_agent, "Arquitectura del Sistema"))
    asyncio.run(generate_document_async(doc_agent, "Plan de Implementación"))
    asyncio.run(generate_document_async(doc_agent, "Plan de Pruebas"))
    asyncio.run(generate_document_async(doc_agent, "BDD Gherkin - Especificaciones"))  # Generar BDD Gherkin
    asyncio.run(generate_document_async(doc_agent, "Plan de Entrega"))

# Iniciar el proyecto
start_project()
