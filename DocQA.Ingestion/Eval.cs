using System.Text.Json;

namespace DocQA.Ingestion;

public record EvalCase(string Id, string Question, string Category, string[] ExpectedFacts, bool ShouldRefuse);

public record EvalResult(
    string Id, string Category, string Question, string Answer,
    bool Refused, bool RefusalCorrect, int FactsHit, int FactsTotal,
    int JudgeScore, string JudgeReasoning);