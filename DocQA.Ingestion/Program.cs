using System.Text;
using System.Text.Json;
using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.Search.Documents.Models;
using DocQA.Ingestion;
using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.Extensions.AI;

var skipIngestion = args.Contains("--query-only");
var runEval = args.Contains("--eval");
var resetIndex = args.Contains("--reset");

var config = new ConfigurationBuilder().AddUserSecrets(System.Reflection.Assembly.GetEntryAssembly()!).Build();
var azureOpenAiEndpoint = config["AzureOpenAI:Endpoint"]!;
var azureOpenAiKey      = config["AzureOpenAI:ApiKey"]!;
var searchEndpoint = config["AzureSearch:Endpoint"]!;
var searchKey      = config["AzureSearch:ApiKey"]!;

const string indexName = "docs";
const int embeddingDimensions = 1536; // text-embedding-3-small

// --- Build the embedding + chat services via Semantic Kernel ---
#pragma warning disable SKEXP0010
var kernel = Kernel.CreateBuilder()
    .AddAzureOpenAIEmbeddingGenerator("embeddings", azureOpenAiEndpoint, azureOpenAiKey)
    .AddAzureOpenAIChatClient("chat", azureOpenAiEndpoint, azureOpenAiKey)
    .Build();
#pragma warning restore SKEXP0010

var embedder = kernel.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
var chat = kernel.GetRequiredService<IChatClient>();
var indexClient = new SearchIndexClient(new Uri(searchEndpoint), new AzureKeyCredential(searchKey));
var searchClient = new SearchClient(new Uri(searchEndpoint), indexName, new AzureKeyCredential(searchKey));

if (resetIndex)
{
    await indexClient.DeleteIndexAsync(indexName, CancellationToken.None);
    Console.WriteLine("Index deleted.");
}

if (runEval) { await RunEvals(); return; }

if (!skipIngestion)
{
    // --- Create (or update) the search index with a vector field ---
    var index = new SearchIndex(indexName)
    {
        Fields =
        {
            new SimpleField("id", SearchFieldDataType.String) { IsKey = true },
            new SearchableField("content"),
            new SimpleField("sourceFile", SearchFieldDataType.String) { IsFilterable = true },
            new SimpleField("chunkIndex", SearchFieldDataType.Int32),
            new SearchField("contentVector", SearchFieldDataType.Collection(SearchFieldDataType.Single))
            {
                IsSearchable = true,
                VectorSearchDimensions = embeddingDimensions,
                VectorSearchProfileName = "vprofile"
            }
        },
        VectorSearch = new VectorSearch
        {
            Profiles = { new VectorSearchProfile("vprofile", "hnsw") },
            Algorithms = { new HnswAlgorithmConfiguration("hnsw") }
        }
    };
    await indexClient.CreateOrUpdateIndexAsync(index);
    Console.WriteLine($"Index '{indexName}' ready.");

    // --- Chunk every markdown file ---
    var docsPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "docs");
    var allChunks = new List<DocumentChunk>();
    foreach (var file in Directory.EnumerateFiles(docsPath, "*.md", SearchOption.AllDirectories))
    {
        var text = await File.ReadAllTextAsync(file);
        if (string.IsNullOrWhiteSpace(text)) continue;
        text = Chunker.MarkdownCleaner.Clean(text);
        if (string.IsNullOrWhiteSpace(text)) continue;
        var rel = Path.GetRelativePath(docsPath, file);
        allChunks.AddRange(Chunker.Chunk(rel, text));
    }

    Console.WriteLine($"Produced {allChunks.Count} chunks from markdown files.");

    // --- Embed in batches and upload ---
    const int batchSize = 50;

    for (var i = 0; i < allChunks.Count; i += batchSize)
    {
        var batch = allChunks.Skip(i).Take(batchSize).ToList();
        var embeddings = await EmbedWithRetry(batch.Select(c => c.Content).ToList());

        var docs = batch.Zip(embeddings, (chunk, vec) => new SearchDocument
        {
            ["id"] = chunk.Id,
            ["content"] = chunk.Content,
            ["sourceFile"] = chunk.SourceFile,
            ["chunkIndex"] = chunk.ChunkIndex,
            ["contentVector"] = vec.Vector.ToArray()
        });

        await searchClient.UploadDocumentsAsync(docs);
        Console.WriteLine($"Uploaded {Math.Min(i + batchSize, allChunks.Count)}/{allChunks.Count}");
        await Task.Delay(2000);
    }

    Console.WriteLine("Ingestion complete.");
}

return;

async Task<GeneratedEmbeddings<Embedding<float>>> EmbedWithRetry(IList<string> inputs)
{
    var delayMs = 5000;
    for (var attempt = 1; attempt <= 6; attempt++)
    {
        try { return await embedder.GenerateAsync(inputs); }
        // Crude 429 detection by message; could inspect a typed status code instead.
        catch (Exception ex) when (ex.Message.Contains("429") && attempt < 6)
        {
            Console.WriteLine($"Rate limited, waiting {delayMs / 1000}s (attempt {attempt})...");
            await Task.Delay(delayMs);
            delayMs = Math.Min(delayMs * 2, 60000); // cap at 60s — the rate window length
        }
    }
    throw new Exception("Embedding failed after retries");
}

async Task<string> Ask(string question)
{
    // 1. Embed the question
    var qVec = (await embedder.GenerateAsync([question]))[0];

    // 2. Retrieve top-k relevant chunks
    var opts = new SearchOptions
    {
        Size = 5,
        VectorSearch = new VectorSearchOptions
        {
            Queries = { new VectorizedQuery(qVec.Vector.ToArray()) { KNearestNeighborsCount = 5, Fields = { "contentVector" } } }
        }
    };
    var hits = await searchClient.SearchAsync<SearchDocument>(null, opts);

    // 3. Assemble the context block from retrieved chunks
    var sb = new StringBuilder();
    var n = 1;
    await foreach (var hit in hits.Value.GetResultsAsync())
    {
        sb.AppendLine($"[Source {n}: {hit.Document["sourceFile"]}]");
        sb.AppendLine(hit.Document["content"].ToString());
        sb.AppendLine();
        n++;
    }
    var context = sb.ToString();

    // 4. Build the grounded prompt — answer only from sources, else refuse, with citations
    var systemPrompt =
        "You are a helpful assistant answering questions about Microsoft Semantic Kernel. " +
        "Answer ONLY using the provided sources below. " +
        "If the sources do not contain the answer, say \"I don't know based on the provided documentation.\" " +
        "Cite the source number(s) you used in brackets, like [Source 1].\n\n" +
        "SOURCES:\n" + context;

    // 5. Call the model
    var reply = await chat.GetResponseAsync(
    [
        new ChatMessage(ChatRole.System, systemPrompt),
        new ChatMessage(ChatRole.User, question)
    ]);
    return reply.Text ?? "(no response)";
}

async Task RunEvals()
{
    var json = await File.ReadAllTextAsync(
        Path.Combine(AppContext.BaseDirectory, "eval-set.json"));
    var cases = JsonSerializer.Deserialize<List<EvalCase>>(json,
        new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;

    var results = new List<EvalResult>();

    foreach (var c in cases)
    {
        Console.WriteLine($"Running {c.Id}...");
        var answer = await Ask(c.Question);

        // Deterministic checks
        var refused = answer.Contains("I don't know", StringComparison.OrdinalIgnoreCase);
        var refusalCorrect = refused == c.ShouldRefuse;
        var factsHit = c.ExpectedFacts.Count(f =>
            answer.Contains(f, StringComparison.OrdinalIgnoreCase));

        // LLM-as-judge
        var (score, reasoning) = await Judge(c, answer);

        results.Add(new EvalResult(c.Id, c.Category, c.Question, answer,
            refused, refusalCorrect, factsHit, c.ExpectedFacts.Length, score, reasoning));
    }

    PrintReport(results);
}

async Task<(int score, string reasoning)> Judge(EvalCase c, string answer)
{
    var judgePrompt =
        "You are grading an answer from a documentation Q&A system about Microsoft Semantic Kernel.\n" +
        $"QUESTION: {c.Question}\n" +
        $"ANSWER: {answer}\n\n" +
        (c.ShouldRefuse
            ? "This question is NOT answerable from Semantic Kernel documentation. " +
              "A CORRECT answer should decline to answer or say it doesn't know. " +
              "Score 5 if it correctly declines, 1 if it fabricated an answer.\n"
            : "Grade the answer's quality on a 1-5 scale: " +
              "5 = accurate, complete, well-grounded; 3 = partially correct or incomplete; " +
              "1 = wrong or irrelevant.\n") +
        "Respond with ONLY a JSON object: {\"score\": <1-5>, \"reasoning\": \"<one sentence>\"}";

    var reply = await chat.GetResponseAsync([new ChatMessage(ChatRole.User, judgePrompt)]);
    var raw = (reply.Text ?? "").Replace("```json", "").Replace("```", "").Trim();

    try
    {
        using var doc = JsonDocument.Parse(raw);
        return (doc.RootElement.GetProperty("score").GetInt32(),
                doc.RootElement.GetProperty("reasoning").GetString() ?? "");
    }
    catch { return (0, $"(judge parse failed: {raw})"); }
}

void PrintReport(List<EvalResult> results)
{
    Console.WriteLine("\n================ EVAL REPORT ================\n");
    foreach (var r in results)
    {
        Console.WriteLine($"{r.Id} [{r.Category}]  judge={r.JudgeScore}/5  " +
            $"facts={r.FactsHit}/{r.FactsTotal}  refusal={(r.RefusalCorrect ? "OK" : "WRONG")}");
        Console.WriteLine($"   Q: {r.Question}");
        Console.WriteLine($"   judge: {r.JudgeReasoning}\n");
    }

    var avgJudge = results.Average(r => r.JudgeScore);
    var refusalAcc = results.Count(r => r.RefusalCorrect) / (double)results.Count;
    Console.WriteLine("---------------------------------------------");
    Console.WriteLine($"Avg judge score:   {avgJudge:F2}/5");
    Console.WriteLine($"Refusal accuracy:  {refusalAcc:P0}");
    foreach (var grp in results.GroupBy(r => r.Category))
        Console.WriteLine($"  {grp.Key,-12} avg judge: {grp.Average(r => r.JudgeScore):F2}");
    Console.WriteLine("=============================================");
}