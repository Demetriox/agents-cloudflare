import { Agent } from 'agents';

export interface Env {
	VECTORIZE: Vectorize;
	AI: Ai;
	// Binding para el Durable Object del Agent (usando el nombre de tu toml)
	MyAgent: DurableObjectNamespace;
	AZURE_OPENAI_ENDPOINT: string;
	AZURE_OPENAI_API_VERSION: string;
	AZURE_OPENAI_DEPLOYMENT_NAME: string;
	AZURE_OPENAI_API_KEY?: string; // This should match your secret name
}

// Updated interface to match Cloudflare AI's actual response type
interface EmbeddingResponse {
	shape?: number[];
	data?: number[][];
	pooling?: "mean" | "cls";
}

// Proper Azure OpenAI response interface
interface AzureOpenAIResponse {
	choices: {
		message: {
			content: string;
		};
		finish_reason: string;
	}[];
	usage?: {
		prompt_tokens: number;
		completion_tokens: number;
		total_tokens: number;
	};
}

class RAGAgent {
	private env: Env;

	constructor(env: Env) {
		this.env = env;
	}

	async shouldUseRAG(question: string): Promise<boolean> {
		// Aquí defines la lógica para determinar si usar RAG
		// Ejemplos de casos donde usar RAG:
		const ragKeywords = [
			'document', 'archivo', 'información específica', 'según el documento',
			'en la base de conocimientos', 'qué dice sobre', 'buscar información',
			'consultar', 'referencias', 'fuentes', 'datos almacenados'
		];

		// Casos donde NO usar RAG (conversación general):
		const generalKeywords = [
			'hola', 'cómo estás', 'ayuda general', 'explicar conceptos',
			'definir', 'cómo funciona', 'qué es', 'ayúdame a entender'
		];

		const questionLower = question.toLowerCase();

		// Verificar si contiene palabras clave para RAG
		const hasRagKeywords = ragKeywords.some(keyword =>
			questionLower.includes(keyword.toLowerCase())
		);

		// Verificar si es una conversación general
		const hasGeneralKeywords = generalKeywords.some(keyword =>
			questionLower.includes(keyword.toLowerCase())
		);

		// Si tiene keywords de RAG, usar RAG
		if (hasRagKeywords) return true;

		// Si tiene keywords generales, no usar RAG
		if (hasGeneralKeywords) return false;

		// Para casos ambiguos, usar un modelo de AI para decidir
		try {
			const decisionPrompt = `Analiza esta pregunta y determina si requiere buscar en una base de conocimientos específica (RAG) o si se puede responder con conocimiento general.

Pregunta: "${question}"

Responde solo con "RAG" si necesita buscar en documentos específicos, o "GENERAL" si se puede responder con conocimiento general.

Respuesta:`;

			const response = await fetch(
				`${this.env.AZURE_OPENAI_ENDPOINT}/openai/deployments/${this.env.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=${this.env.AZURE_OPENAI_API_VERSION}`,
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						"api-key": this.env.AZURE_OPENAI_API_KEY || "", // Fixed: Direct access to env variable
					},
					body: JSON.stringify({
						messages: [
							{
								role: "system",
								content: "Decide si la pregunta requiere RAG (documentos) o GENERAL (conocimiento común). Responde solo 'RAG' o 'GENERAL'."
							},
							{
								role: "user",
								content: `Pregunta: "${question}"`
							}
						],
						temperature: 0.7,
						max_tokens: 10,
						top_p: 0.9
					}),
				}
			);

			if (!response.ok) {
				throw new Error(`Azure OpenAI API error: ${response.status} - ${await response.text()}`);
			}

			const decisionResponse: AzureOpenAIResponse = await response.json(); // Fixed: Parse JSON first
			const content = decisionResponse.choices?.[0]?.message?.content || "";

			return content.toLowerCase().includes("rag");
		} catch (error) {
			console.error("Error in shouldUseRAG:", error);
			// En caso de error, defaultear a usar RAG si la pregunta es específica
			return questionLower.length > 20; // Preguntas más largas tienden a ser más específicas
		}
	}

	async processWithRAG(question: string, topK = 3): Promise<any> {
		try {
			// Step 1: Generate embedding for the question
			const queryVector = await this.env.AI.run(
				"@cf/baai/bge-base-en-v1.5",
				{ text: [question] }
			) as EmbeddingResponse;

			if (!queryVector.data || queryVector.data.length === 0) {
				throw new Error("Failed to generate query embedding");
			}

			// Step 2: Search for relevant documents
			const matches = await this.env.VECTORIZE.query(queryVector.data[0], {
				topK,
				returnMetadata: true,
			});

			// Step 3: Extract relevant context from matched documents
			const context = matches.matches
				.filter(match => match.score > 0.7) // Filter by similarity threshold
				.map(match => match.metadata?.content)
				.filter(content => content)
				.join('\n\n');

			// Step 4: Generate response using LLM with context
			let answer = "No pude encontrar información relevante en la base de conocimientos.";

			if (context.length > 0) {
				const prompt = `Context information:
${context}

Question: ${question}

Please provide a helpful and accurate answer based on the context information above. If the context doesn't contain enough information to answer the question, please say so.

Answer:`;

				const response = await fetch(
					`${this.env.AZURE_OPENAI_ENDPOINT}/openai/deployments/${this.env.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=${this.env.AZURE_OPENAI_API_VERSION}`,
					{
						method: "POST",
						headers: {
							"Content-Type": "application/json",
							"api-key": this.env.AZURE_OPENAI_API_KEY || "", // Fixed: Direct access
						},
						body: JSON.stringify({
							messages: [
								{
									role: "system",
									content: "Eres un asistente útil y creativo. Responde en español de forma clara y detallada basándote en el contexto proporcionado."
								},
								{
									role: "user",
									content: prompt // Fixed: Use the proper prompt with context
								}
							],
							temperature: 0.7,
							max_tokens: 300,
							top_p: 0.9
						}),
					}
				);

				if (!response.ok) {
					throw new Error(`Azure OpenAI API error: ${response.status} - ${await response.text()}`);
				}

				const chatResponse: AzureOpenAIResponse = await response.json(); // Fixed: Parse JSON first
				answer = chatResponse.choices?.[0]?.message?.content || "No pude generar una respuesta.";
			}

			return {
				question,
				answer,
				usedRAG: true,
				sources: matches.matches.map(match => ({
					id: match.id,
					score: match.score,
					title: match.metadata?.title,
					content: typeof match.metadata?.content === 'string'
						? match.metadata.content.substring(0, 200) + "..."
						: undefined
				})),
				context_used: context.length > 0
			};

		} catch (error) {
			console.error("Error in processWithRAG:", error);
			throw new Error(`RAG processing failed: ${error instanceof Error ? error.message : "Unknown error"}`);
		}
	}

	async processGeneral(question: string): Promise<any> {
		try {
			const response = await fetch(
				`${this.env.AZURE_OPENAI_ENDPOINT}/openai/deployments/${this.env.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=${this.env.AZURE_OPENAI_API_VERSION}`,
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						"api-key": this.env.AZURE_OPENAI_API_KEY || "", // Fixed: Direct access
					},
					body: JSON.stringify({
						messages: [
							{
								role: "system",
								content: "Eres un asistente útil y creativo. Responde en español de forma clara y detallada." // Fixed: Proper system message
							},
							{
								role: "user",
								content: question // Fixed: Use the actual question
							}
						],
						temperature: 0.7,
						max_tokens: 300, // Fixed: Increased token limit
						top_p: 0.9
					}),
				}
			);

			if (!response.ok) {
				throw new Error(`Azure OpenAI API error: ${response.status} - ${await response.text()}`);
			}

			const chatResponse: AzureOpenAIResponse = await response.json(); // Fixed: Parse JSON first
			const answer = chatResponse.choices?.[0]?.message?.content || "No pude generar una respuesta.";

			return {
				question,
				answer,
				usedRAG: false,
				sources: [],
				context_used: false
			};

		} catch (error) {
			console.error("Error in processGeneral:", error);
			throw new Error(`General processing failed: ${error instanceof Error ? error.message : "Unknown error"}`);
		}
	}
}

export default {
	async fetch(request, env, ctx): Promise<Response> {
		let path = new URL(request.url).pathname;

		if (path.startsWith("/favicon")) {
			return new Response("", { status: 404 });
		}

		// Add CORS headers for browser requests
		const corsHeaders = {
			'Access-Control-Allow-Origin': '*',
			'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
			'Access-Control-Allow-Headers': 'Content-Type',
		};

		if (request.method === 'OPTIONS') {
			return new Response(null, { headers: corsHeaders });
		}

		// Initialize the RAG Agent
		const agent = new RAGAgent(env);

		// Enhanced chat endpoint with Agent decision-making
		if (path === "/chat" && request.method === "POST") {
			try {
				const { question, topK = 3, forceRAG = false } = await request.json() as {
					question: string;
					topK?: number;
					forceRAG?: boolean;
				};

				if (!question) {
					return Response.json({
						error: "Question is required"
					}, {
						status: 400,
						headers: corsHeaders
					});
				}

				// Let the agent decide whether to use RAG or not
				const shouldUseRAG = forceRAG || await agent.shouldUseRAG(question);

				let result;
				if (shouldUseRAG) {
					result = await agent.processWithRAG(question, topK);
				} else {
					result = await agent.processGeneral(question);
				}

				// Guardar la interacción en el Durable Object para historial
				try {
					const agentId = env.MyAgent.idFromName("main-agent");
					const agentStub = env.MyAgent.get(agentId);

					await agentStub.fetch("https://agent.internal/save", {
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({
							question: result.question,
							answer: result.answer,
							usedRAG: result.usedRAG,
							timestamp: new Date().toISOString()
						})
					});
				} catch (error) {
					// No fallar si no se puede guardar el historial
					console.error("Failed to save to agent history:", error);
				}

				return Response.json({
					...result,
					agentDecision: shouldUseRAG ? "RAG" : "GENERAL",
					timestamp: new Date().toISOString()
				}, { headers: corsHeaders });

			} catch (error) {
				console.error("Chat endpoint error:", error);
				return Response.json({
					error: "Failed to process chat request",
					details: error instanceof Error ? error.message : "Unknown error"
				}, {
					status: 500,
					headers: corsHeaders
				});
			}
		}

		// Insert documents with embeddings (Knowledge Base)
		if (path === "/insert" && request.method === "POST") {
			try {
				const { documents } = await request.json() as { documents: any[] };

				if (!documents || !Array.isArray(documents)) {
					return Response.json({
						error: "Documents array is required"
					}, {
						status: 400,
						headers: corsHeaders
					});
				}

				// Generate embeddings for all documents
				const texts = documents.map(doc => doc.content || doc.text || doc);
				const modelResp = await env.AI.run(
					"@cf/baai/bge-base-en-v1.5",
					{ text: texts }
				) as EmbeddingResponse;

				if (!modelResp.data) {
					return Response.json({
						error: "No embedding data received"
					}, {
						status: 500,
						headers: corsHeaders
					});
				}

				// Create vectors with metadata
				let vectors: VectorizeVector[] = [];
				modelResp.data.forEach((vector, index) => {
					const doc = documents[index];
					vectors.push({
						id: `doc-${Date.now()}-${index}`,
						values: vector,
						metadata: {
							content: typeof doc === 'string' ? doc : doc.content || doc.text,
							title: typeof doc === 'object' ? doc.title : `Document ${index + 1}`,
							source: typeof doc === 'object' ? doc.source : 'manual_insert',
							timestamp: new Date().toISOString(),
							type: 'document'
						}
					});
				});

				const inserted = await env.VECTORIZE.upsert(vectors);

				return Response.json({
					success: true,
					inserted: vectors.length,
					message: `Successfully inserted ${vectors.length} documents`
				}, { headers: corsHeaders });

			} catch (error) {
				return Response.json({
					error: "Failed to insert documents",
					details: error instanceof Error ? error.message : "Unknown error"
				}, {
					status: 500,
					headers: corsHeaders
				});
			}
		}

		// Search documents endpoint
		if (path === "/search" && request.method === "POST") {
			try {
				const { query, topK = 5 } = await request.json() as { query: string; topK?: number };

				if (!query) {
					return Response.json({
						error: "Query is required"
					}, {
						status: 400,
						headers: corsHeaders
					});
				}

				const queryVector = await env.AI.run(
					"@cf/baai/bge-base-en-v1.5",
					{ text: [query] }
				) as EmbeddingResponse;

				if (!queryVector.data || queryVector.data.length === 0) {
					return Response.json({
						error: "No query vector data received"
					}, {
						status: 500,
						headers: corsHeaders
					});
				}

				const matches = await env.VECTORIZE.query(queryVector.data[0], {
					topK,
					returnMetadata: true,
				});

				return Response.json({
					query,
					matches: matches.matches.map(match => ({
						id: match.id,
						score: match.score,
						title: match.metadata?.title,
						content: match.metadata?.content,
						source: match.metadata?.source,
						timestamp: match.metadata?.timestamp
					}))
				}, { headers: corsHeaders });

			} catch (error) {
				return Response.json({
					error: "Search failed",
					details: error instanceof Error ? error.message : "Unknown error"
				}, {
					status: 500,
					headers: corsHeaders
				});
			}
		}

		// Agent status endpoint
		if (path === "/agent/status") {
			return Response.json({
				status: "Agent is running",
				capabilities: ["RAG decision-making", "Document search", "General conversation"],
				timestamp: new Date().toISOString()
			}, { headers: corsHeaders });
		}

		// Agent history endpoint
		if (path === "/agent/history") {
			try {
				const agentId = env.MyAgent.idFromName("main-agent");
				const agentStub = env.MyAgent.get(agentId);
				const response = await agentStub.fetch("https://agent.internal/history");
				const data = await response.json();

				return Response.json(data, { headers: corsHeaders });
			} catch (error) {
				return Response.json({
					error: "Failed to get agent history",
					details: error instanceof Error ? error.message : "Unknown error"
				}, { status: 500, headers: corsHeaders });
			}
		}

		// Agent statistics endpoint
		if (path === "/agent/stats") {
			try {
				const agentId = env.MyAgent.idFromName("main-agent");
				const agentStub = env.MyAgent.get(agentId);
				const response = await agentStub.fetch("https://agent.internal/stats");
				const data = await response.json();

				return Response.json(data, { headers: corsHeaders });
			} catch (error) {
				return Response.json({
					error: "Failed to get agent stats",
					details: error instanceof Error ? error.message : "Unknown error"
				}, { status: 500, headers: corsHeaders });
			}
		}

		// Test agent decision endpoint
		if (path === "/agent/test" && request.method === "POST") {
			try {
				const { question } = await request.json() as { question: string };

				if (!question) {
					return Response.json({
						error: "Question is required"
					}, {
						status: 400,
						headers: corsHeaders
					});
				}

				const shouldUseRAG = await agent.shouldUseRAG(question);

				return Response.json({
					question,
					decision: shouldUseRAG ? "RAG" : "GENERAL",
					reasoning: shouldUseRAG
						? "This question appears to require specific document search"
						: "This question can be answered with general knowledge"
				}, { headers: corsHeaders });

			} catch (error) {
				return Response.json({
					error: "Failed to test agent decision",
					details: error instanceof Error ? error.message : "Unknown error"
				}, {
					status: 500,
					headers: corsHeaders
				});
			}
		}

		// Health check endpoint
		if (path === "/health") {
			return Response.json({
				status: "healthy",
				services: ["RAG", "Agent", "Vectorize", "Workers AI"],
				timestamp: new Date().toISOString()
			}, { headers: corsHeaders });
		}

		// Default route - show available endpoints
		return Response.json({
			message: "RAG API Server with Intelligent Agent",
			endpoints: {
				"POST /chat": "Ask questions with intelligent RAG/General routing",
				"POST /insert": "Insert documents into knowledge base",
				"POST /search": "Search documents by similarity",
				"POST /agent/test": "Test agent decision making",
				"GET /agent/status": "Get agent status",
				"GET /agent/history": "Get conversation history from Durable Object",
				"GET /agent/stats": "Get agent usage statistics",
				"GET /health": "Health check"
			},
			features: [
				"Intelligent routing between RAG and general responses",
				"Document embedding and search",
				"AI-powered decision making",
				"Vector similarity search",
				"Persistent conversation history via Durable Objects",
				"Usage analytics and statistics"
			]
		}, { headers: corsHeaders });
	},
} satisfies ExportedHandler<Env>;

// Durable Object para funcionalidades avanzadas del Agent
export class MyAgent implements DurableObject {
	private storage: DurableObjectStorage;
	private env: Env;

	constructor(state: DurableObjectState, env: Env) {
		this.storage = state.storage;
		this.env = env;
	}

	async fetch(request: Request): Promise<Response> {
		const url = new URL(request.url);
		const path = url.pathname;

		// CORS headers
		const corsHeaders = {
			'Access-Control-Allow-Origin': '*',
			'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
			'Access-Control-Allow-Headers': 'Content-Type',
		};

		if (request.method === 'OPTIONS') {
			return new Response(null, { headers: corsHeaders });
		}

		// Mantener historial de conversaciones
		if (path === "/history" && request.method === "GET") {
			try {
				const history = await this.storage.get("chat_history") || [];
				return Response.json({ history }, { headers: corsHeaders });
			} catch (error) {
				return Response.json({
					error: "Failed to retrieve history",
					details: error instanceof Error ? error.message : "Unknown error"
				}, { status: 500, headers: corsHeaders });
			}
		}

		// Guardar interacción en el historial
		if (path === "/save" && request.method === "POST") {
			try {
				const { question, answer, usedRAG, timestamp } = await request.json() as {
					question: string;
					answer: string;
					usedRAG: boolean;
					timestamp: string;
				};

				let history: any[] = await this.storage.get("chat_history") || [];
				history.push({ question, answer, usedRAG, timestamp });

				// Mantener solo los últimos 100 intercambios
				if (history.length > 100) {
					history = history.slice(-100);
				}

				await this.storage.put("chat_history", history);

				return Response.json({
					success: true,
					historyLength: history.length
				}, { headers: corsHeaders });
			} catch (error) {
				return Response.json({
					error: "Failed to save to history",
					details: error instanceof Error ? error.message : "Unknown error"
				}, { status: 500, headers: corsHeaders });
			}
		}

		// Estadísticas del Agent
		if (path === "/stats" && request.method === "GET") {
			try {
				const history: any[] = await this.storage.get("chat_history") || [];
				const ragUsage = history.filter(item => item.usedRAG).length;
				const generalUsage = history.filter(item => !item.usedRAG).length;

				return Response.json({
					totalInteractions: history.length,
					ragUsage,
					generalUsage,
					ragPercentage: history.length > 0 ? Math.round((ragUsage / history.length) * 100) : 0
				}, { headers: corsHeaders });
			} catch (error) {
				return Response.json({
					error: "Failed to get stats",
					details: error instanceof Error ? error.message : "Unknown error"
				}, { status: 500, headers: corsHeaders });
			}
		}

		return Response.json({
			error: "Not found"
		}, { status: 404, headers: corsHeaders });
	}
}