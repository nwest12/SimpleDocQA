using System.Text;
using System.Text.RegularExpressions;

namespace DocQA.Ingestion;

public record DocumentChunk(string Id, string SourceFile, int ChunkIndex, string Content);

public static partial class Chunker
{
    // Defaults. Pass explicit values to experiment with different sizes.
    public const int DefaultTargetChars  = 2000;   // ~500 tokens
    public const int DefaultOverlapChars = 200;     // ~50 tokens

    public static IEnumerable<DocumentChunk> Chunk(
        string sourceFile,
        string text,
        int targetChars  = DefaultTargetChars,
        int overlapChars = DefaultOverlapChars)
    {
        var paragraphs = text.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
        var current = new StringBuilder();
        int index = 0;

        foreach (var para in paragraphs)
        {
            if (current.Length + para.Length > targetChars && current.Length > 0)
            {
                yield return Make(sourceFile, index++, current.ToString());
                var tail = current.ToString();
                current.Clear();
                current.Append(tail[^Math.Min(overlapChars, tail.Length)..]);
            }
            current.Append(para).Append("\n\n");
        }
        if (current.Length > 0)
            yield return Make(sourceFile, index, current.ToString());
    }

    private static DocumentChunk Make(string file, int i, string content)
    {
        // AI Search keys allow only letters, digits, dash, underscore, equals
        var safeBase = string.Concat(
            Path.GetFileNameWithoutExtension(file)
                .Select(c => char.IsLetterOrDigit(c) ? c : '_'));
        return new DocumentChunk($"{safeBase}-{i}", file, i, content.Trim());
    }
    
    public static partial class MarkdownCleaner
    {
        public static string Clean(string text)
        {
            // 1. Strip YAML frontmatter (--- ... --- at the very top)
            text = FrontmatterRegex().Replace(text, "");

            // 2. Remove DocFX zone pivot markers (::: zone pivot="..." and ::: zone-end)
            text = ZoneRegex().Replace(text, "");

            // 3. Remove ms doc-link paths like (/dotnet/api/microsoft.semantickernel...)
            //    Keep the link TEXT, drop the (/path) target
            text = MsLinkRegex().Replace(text, "$1");

            // 4. Strip MS-specific alert/note syntax markers (> [!NOTE], > [!TIP], etc.)
            text = AlertRegex().Replace(text, "");

            // 5. Collapse 3+ blank lines into a clean paragraph break
            text = BlankLinesRegex().Replace(text, "\n\n");

            return text.Trim();
        }

        [GeneratedRegex(@"^---\s*\n.*?\n---\s*\n", RegexOptions.Singleline)]
        private static partial Regex FrontmatterRegex();

        [GeneratedRegex(@"::: *zone[^\n]*\n?")]
        private static partial Regex ZoneRegex();

        [GeneratedRegex(@"\[([^\]]+)\]\(/[^)]+\)")]
        private static partial Regex MsLinkRegex();

        [GeneratedRegex(@"> *\[!\w+\]\n?")]
        private static partial Regex AlertRegex();

        [GeneratedRegex(@"\n{3,}")]
        private static partial Regex BlankLinesRegex();
    }
}