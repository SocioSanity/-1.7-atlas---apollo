#include "atlas_orchestrator.h"
#include "atlas_model_bridge.h"
#include "atlas_conversation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>

static AtlasSpecialist g_specialists[ATLAS_MAX_SPECIALISTS];
static int g_specialist_count = 0;

static void load_feedback(AtlasSpecialist *specialist)
{
    FILE *f = fopen("Atlas_1.0.learning.tsv", "rb");
    if (!f) return;
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        char *first = strchr(line, '\t');
        if (!first) continue;
        *first++ = 0;
        char *second = strchr(first, '\t');
        if (!second) continue;
        *second++ = 0;
        if (strcmp(line, specialist->model) != 0) continue;
        specialist->successful_uses = strtoull(first, NULL, 10);
        specialist->useful_uses = strtoull(second, NULL, 10);
    }
    fclose(f);
}

static void record_session_feedback(const AtlasReasoningSession *session)
{
    FILE *f = fopen("Atlas_1.0.learning.tsv", "ab");
    for (int i = 0; i < session->task_count; ++i) {
        const AtlasSubtask *task = &session->tasks[i];
        for (int r = 0; r < task->response_count; ++r) {
            int index = task->response_specialist_indices[r];
            if (index < 0 || index >= g_specialist_count) continue;
            AtlasSpecialist *s = &g_specialists[index];
            ++s->successful_uses;
            if (session->verified) ++s->useful_uses;
            if (f) fprintf(f, "%s\t%llu\t%llu\n", s->model,
                (unsigned long long)s->successful_uses,
                (unsigned long long)s->useful_uses);
        }
    }
    if (f) fclose(f);
}

static int contains_ci(const char *text, const char *needle)
{
    if (!text || !needle || !*needle) return 0;
    size_t n = strlen(needle);
    for (const char *p = text; *p; ++p) {
        size_t i = 0;
        while (i < n && p[i] &&
               tolower((unsigned char)p[i]) == tolower((unsigned char)needle[i]))
            ++i;
        if (i == n) return 1;
    }
    return 0;
}

static int verification_passes(const char *text)
{
    if (!text) return 0;
    while (*text && isspace((unsigned char)*text)) ++text;
    if (strncmp(text, "VERIFIED:", 9) != 0) return 0;
    static const char *defect_phrases[] = {
        "REJECTED:", "logical error in", "has a logical error",
        "contains a logical error", "incorrect", "unsupported",
        "missing requirement", "unresolved issue", "unresolved error",
        "contradiction", "cannot compile", "does not satisfy", "fails to"
    };
    for (size_t i = 0; i < sizeof(defect_phrases) / sizeof(defect_phrases[0]); ++i)
        if (contains_ci(text, defect_phrases[i])) return 0;
    return 1;
}

static int asks_for_implementation(const char *problem)
{
    if (!problem || contains_ci(problem,"do not write source") ||
        contains_ci(problem,"don't write source") ||
        contains_ci(problem,"do not write code") ||
        contains_ci(problem,"don't write code") ||
        contains_ci(problem,"no code yet") ||
        contains_ci(problem,"without writing code")) return 0;
    if (contains_ci(problem,"function") &&
        (contains_ci(problem,"write") || contains_ci(problem,"implement") ||
         contains_ci(problem,"create") || contains_ci(problem,"provide") ||
         contains_ci(problem,"generate"))) return 1;
    if (contains_ci(problem,"program") &&
        (contains_ci(problem,"write") || contains_ci(problem,"implement") ||
         contains_ci(problem,"create") || contains_ci(problem,"generate"))) return 1;
    return contains_ci(problem,"write code") || contains_ci(problem,"write a program") ||
           contains_ci(problem,"write a c program") || contains_ci(problem,"write c code") ||
           contains_ci(problem,"write a function") || contains_ci(problem,"write source code") ||
           contains_ci(problem,"write a complete") || contains_ci(problem,"compile-ready") ||
           contains_ci(problem,"complete c source") || contains_ci(problem,"provide source code") ||
           contains_ci(problem,"implement a ") || contains_ci(problem,"implement the ") ||
           contains_ci(problem,"implement this") || contains_ci(problem,"implement it") ||
           contains_ci(problem,"full source") || contains_ci(problem,"c implementation");
}

static int answer_has_code(const char *answer)
{
    return contains_ci(answer,"```") || contains_ci(answer,"#include") ||
           contains_ci(answer,"int main(") || contains_ci(answer,"int main (");
}

static int answer_meets_basic_code_requirements(const char *problem, const char *answer)
{
    if (!asks_for_implementation(problem)) return 1;
    if (!answer_has_code(answer)) return 0;
    if (contains_ci(problem,"size_t") && !contains_ci(answer,"size_t")) return 0;
    if (contains_ci(problem,"overflow") &&
        !contains_ci(answer,"SIZE_MAX") && !contains_ci(answer,"ERANGE")) return 0;
    if (contains_ci(problem,"trailing character") &&
        !contains_ci(answer,"endptr") && !contains_ci(answer,"*end") &&
        !contains_ci(answer,"strtou")) return 0;
    if (contains_ci(problem,"parse_size") && !contains_ci(answer,"parse_size")) return 0;
    if (contains_ci(problem,"empty") && !contains_ci(answer,"!*text") &&
        !contains_ci(answer,"*text ==") && !contains_ci(answer,"text[0] ==")) return 0;
    if ((contains_ci(problem,"nondigit") || contains_ci(problem,"non-digit") ||
         contains_ci(problem,"signs")) && !contains_ci(answer,"isdigit") &&
        !contains_ci(answer,"<'0'") && !contains_ci(answer,">'9'")) return 0;
    if (contains_ci(problem,"unchanged") && !contains_ci(answer,"*out =")) return 0;
    if (contains_ci(problem,"no main") &&
        (contains_ci(answer,"main(") || contains_ci(answer,"main ("))) return 0;
    if (contains_ci(problem,"return 1") && contains_ci(problem,"return 0") &&
        (!contains_ci(answer,"return 1") || !contains_ci(answer,"return 0") ||
         contains_ci(answer,"return -1"))) return 0;
    if (contains_ci(problem,"SIZE_MAX") && !contains_ci(answer,"SIZE_MAX")) return 0;
    if (contains_ci(problem,"realloc") && !contains_ci(answer,"realloc")) return 0;
    if (contains_ci(problem,"push") && !contains_ci(answer,"push")) return 0;
    if (contains_ci(problem,"pop") && !contains_ci(answer,"pop")) return 0;
    if (contains_ci(problem,"peek") && !contains_ci(answer,"peek")) return 0;
    if (contains_ci(problem,"main") &&
        !contains_ci(answer,"main(") && !contains_ci(answer,"main (")) return 0;
    return 1;
}

static int append_format(char *dst, size_t capacity, size_t *used,
                         const char *format, ...)
{
    if (!dst || !used || *used >= capacity) return 0;
    va_list args;
    va_start(args, format);
    int n = vsnprintf(dst + *used, capacity - *used, format, args);
    va_end(args);
    if (n < 0) return 0;
    if ((size_t)n >= capacity - *used) {
        *used = capacity - 1;
        dst[*used] = 0;
        return 0;
    }
    *used += (size_t)n;
    return 1;
}

static void safe_copy(char *dst, size_t size, const char *src)
{
    if (!dst || size == 0)
        return;

    if (!src)
        src = "";

    snprintf(dst, size, "%s", src);
}

void atlas_clear_specialists(void)
{
    memset(g_specialists, 0, sizeof(g_specialists));
    g_specialist_count = 0;
}

int atlas_register_specialist(
    const char *name,
    const char *model,
    float reasoning,
    float coding,
    float research,
    float mathematics,
    float analysis,
    float speed)
{
    if (!name || !model)
        return 1;

    if (g_specialist_count >= ATLAS_MAX_SPECIALISTS)
        return 2;

    AtlasSpecialist *s = &g_specialists[g_specialist_count];

    memset(s, 0, sizeof(*s));

    safe_copy(s->name, sizeof(s->name), name);
    safe_copy(s->model, sizeof(s->model), model);

    s->reasoning = reasoning;
    s->coding = coding;
    s->research = research;
    s->mathematics = mathematics;
    s->analysis = analysis;
    s->speed = speed;
    s->enabled = 1;
    load_feedback(s);

    ++g_specialist_count;

    return 0;
}

static int ask(
    const char *exe,
    const char *model,
    const char *prompt,
    char *response,
    size_t response_size)
{
    unsigned budget = 128;
    if (contains_ci(prompt, "Reply exactly COMPLETE")) budget = 48;
    else if (contains_ci(prompt, "Return one task per line")) budget = 96;
    else if (contains_ci(prompt, "Return either:") ||
             contains_ci(prompt, "Return exactly VERIFIED:")) budget = 64;
    else if (contains_ci(prompt, "You are a specialist assisting"))
        budget = asks_for_implementation(prompt) ? 384 : 128;
    else if (contains_ci(prompt, "larger reference model")) budget = 512;
    else if (contains_ci(prompt, "Now produce the best unified answer") ||
             contains_ci(prompt, "Produce one corrected answer"))
        budget = asks_for_implementation(prompt) ? 512 : 192;
    return atlas_model_ask_with_budget(
        exe,
        model,
        prompt,
        budget,
        response,
        response_size
    );
}

static int ask_specialist(const char *exe,const AtlasSpecialist *specialist,
                          int code_request,const char *prompt,
                          char *response,size_t response_size)
{
    if(!specialist)return 1;
    unsigned budget=code_request?384:128;
    if(code_request&&specialist->speed>0.0f&&specialist->speed<0.5f)
        budget=192;
    return atlas_model_ask_with_budget(exe,specialist->model,prompt,budget,
                                        response,response_size);
}

static int parse_decomposition(
    const char *text,
    AtlasReasoningSession *session)
{
    if (!text || !session)
        return 1;

    /*
     * Expected format:
     *
     * TASK 1: ...
     * TASK 2: ...
     * TASK 3: ...
     *
     * The parser deliberately accepts loose model output.
     */

    const char *p = text;

    while (*p && session->task_count < 4) {

        const char *task = strstr(p, "TASK ");

        if (!task)
            break;

        const char *colon = strchr(task, ':');

        if (!colon)
            break;

        AtlasSubtask *sub = &session->tasks[session->task_count];

        memset(sub, 0, sizeof(*sub));

        sub->id = session->task_count + 1;

        ++colon;

        while (*colon == ' ' || *colon == '\t')
            ++colon;

        const char *end = strchr(colon, '\n');

        size_t len;

        if (end)
            len = (size_t)(end - colon);
        else
            len = strlen(colon);

        if (len >= sizeof(sub->description))
            len = sizeof(sub->description) - 1;

        memcpy(
            sub->description,
            colon,
            len
        );

        sub->description[len] = '\0';

        ++session->task_count;

        if (!end)
            break;

        p = end + 1;
    }

    int invalid = session->task_count == 0;
    for (int i = 0; i < session->task_count && !invalid; ++i) {
        const char *description = session->tasks[i].description;
        if (strlen(description) < 24) invalid = 1;
        for (int j = 0; j < i; ++j)
            if (strcmp(description, session->tasks[j].description) == 0)
                invalid = 1;
    }

    /*
     * Tiny placeholders and duplicate tasks are not useful decomposition.
     * Give one specialist the full problem in that case.
     */
    if (invalid) {
        memset(session->tasks, 0, sizeof(session->tasks));
        session->tasks[0].id = 1;
        safe_copy(session->tasks[0].description,
                  sizeof(session->tasks[0].description),
                  session->original_problem);
        session->task_count = 1;
    }

    return 0;
}

static float specialist_score(
    const AtlasSpecialist *s,
    const char *task)
{
    if (!s || !s->enabled)
        return -1.0f;

    /*
     * Initial capability scoring.
     *
     * This intentionally remains simple. Atlas's model
     * performs the actual semantic routing; these values
     * provide a capability prior for selecting candidates.
     */

    float score = s->reasoning +
                  s->coding +
                  s->research +
                  s->mathematics +
                  s->analysis;

    /* Size is a modest tie-breaker. Validation results and task fit should
       outrank the fastest model when the quality gap is meaningful. */
    if (s->speed > 0.0f)
        score += s->speed * 0.5f;

    if (s->successful_uses) {
        float observed = (float)s->useful_uses / (float)s->successful_uses;
        score += (observed - 0.5f) * 2.0f;
    }

    /*
     * Small keyword hints provide deterministic tie-breaking.
     */
    if (task) {

        if (contains_ci(task, "code") ||
            contains_ci(task, "programming") ||
            contains_ci(task, "C ") ||
            contains_ci(task, "bug") ||
            contains_ci(task, "compile") ||
            contains_ci(task, "program") ||
            contains_ci(task, "implement") ||
            contains_ci(task, "function") ||
            contains_ci(task, "algorithm") ||
            contains_ci(task, "source code") ||
            contains_ci(task, "realloc") ||
            contains_ci(task, "stack")) {

            score += s->coding * 5.0f;
        }

        if (contains_ci(task, "math") ||
            contains_ci(task, "calculate") ||
            contains_ci(task, "equation")) {

            score += s->mathematics * 2.0f;
        }

        if (contains_ci(task, "research") ||
            contains_ci(task, "information") ||
            contains_ci(task, "knowledge")) {

            score += s->research * 2.0f;
        }

        if (contains_ci(task, "reason") ||
            contains_ci(task, "why") ||
            contains_ci(task, "analyze") ||
            contains_ci(task, "analyse")) {

            score += s->reasoning * 2.0f;
        }
    }

    return score;
}

static int same_model_path(const char *a, const char *b)
{
    if (!a || !b) return 0;
    while (*a && *b) {
        char ca = *a == '/' ? '\\' : *a;
        char cb = *b == '/' ? '\\' : *b;
        if (tolower((unsigned char)ca) != tolower((unsigned char)cb)) return 0;
        ++a;
        ++b;
    }
    return *a == *b;
}

static void choose_specialists(
    const char *exe,
    const char *router_model,
    AtlasSubtask *task)
{
    if (!exe || !router_model || !task)
        return;

    task->specialist_count = 0;

    float best_score[8];

    for (int i = 0; i < 8; ++i)
        best_score[i] = -1.0f;

    for (int i = 0; i < g_specialist_count; ++i) {

        /* Small models are routers; reserve actual answer generation for
           their larger specialists. */
        if (same_model_path(g_specialists[i].model, router_model) ||
            g_specialists[i].speed >= 1.5f)
            continue;

        float score = specialist_score(
            &g_specialists[i],
            task->description
        );

        if (score < 0.0f)
            continue;

        for (int slot = 0; slot < 8; ++slot) {

            if (score > best_score[slot]) {

                for (int move = 7; move > slot; --move) {
                    best_score[move] = best_score[move - 1];

                    if (move > 0)
                        task->specialist_indices[move] =
                            task->specialist_indices[move - 1];
                }

                best_score[slot] = score;

                task->specialist_indices[slot] = i;

                if (task->specialist_count < slot + 1)
                    task->specialist_count = slot + 1;

                break;
            }
        }
    }

    /* Keep the fast candidates first and reserve one late slot for the
       largest compatible model, which acts as an escalation path. */
    if(task->specialist_count>0){
        int small[8],large[8],ordered[8];int ns=0,nl=0,no=0;
        for(int i=0;i<task->specialist_count;++i){
            int index=task->specialist_indices[i];
            if(index<0||index>=g_specialist_count)continue;
            if(g_specialists[index].speed>0.0f&&g_specialists[index].speed<0.5f)
                large[nl++]=index;
            else small[ns++]=index;
        }
        for(int i=0;i<ns&&i<3;++i)ordered[no++]=small[i];
        if(nl&&no<8)ordered[no++]=large[0];
        for(int i=3;i<ns&&no<4;++i)ordered[no++]=small[i];
        for(int i=1;i<nl&&no<4;++i)ordered[no++]=large[i];
        for(int i=0;i<no;++i)task->specialist_indices[i]=ordered[i];
        task->specialist_count=no;
    }

    /* Ask the small domain router to choose among the compatible, larger
       candidates. Keep the scored order if its reply is invalid. */
    if (task->specialist_count > 1) {
        char route_prompt[4096];
        char route_reply[128] = {0};
        size_t used = 0;
        (void)append_format(route_prompt, sizeof(route_prompt), &used,
            "You are Atlas's specialist router. Choose the best model for this task; do not solve it.\n"
            "Reply with only its candidate number.\n\nTASK:\n%.1800s\n\nCANDIDATES:\n",
            task->description);
        for (int i = 0; i < task->specialist_count; ++i) {
            int index = task->specialist_indices[i];
            if (index < 0 || index >= g_specialist_count) continue;
            if (!append_format(route_prompt, sizeof(route_prompt), &used,
                               "%d. %s\n", i + 1, g_specialists[index].name))
                break;
        }
        if (atlas_model_ask_with_budget(exe, router_model, route_prompt, 24,
                                        route_reply, sizeof(route_reply)) == 0) {
            char *end = route_reply;
            while (*end && !isdigit((unsigned char)*end)) ++end;
            if (*end) {
                long selected = strtol(end, NULL, 10);
                if (selected >= 1 && selected <= task->specialist_count && selected != 1) {
                    int chosen = task->specialist_indices[selected - 1];
                    for (int i = (int)selected - 1; i > 0; --i)
                        task->specialist_indices[i] = task->specialist_indices[i - 1];
                    task->specialist_indices[0] = chosen;
                }
            }
        }
    }

    task->duplicate_count = 1;
}

static int collect_specialist_answers(
    const char *exe,
    const char *router_model,
    const char *original_problem,
    AtlasSubtask *task)
{
    if (!exe || !router_model || !original_problem || !task)
        return 1;

    task->response_count = 0;

    /* Begin with the highest-ranked model. Ask the small coordinator if
       another specialist is needed; every inference process exits before
       the next model is started. */
    int max_attempts = task->specialist_count < 4 ? task->specialist_count : 4;

    for (int specialist_slot = 0;
         specialist_slot < max_attempts &&
         task->response_count < 4;
         ++specialist_slot) {

        int specialist_index =
            task->specialist_indices[specialist_slot];

        if (specialist_index < 0 ||
            specialist_index >= g_specialist_count)
            continue;

        AtlasSpecialist *s =
            &g_specialists[specialist_index];

        char prompt[ATLAS_MAX_TEXT];

        const int code_request=asks_for_implementation(original_problem);
        snprintf(prompt,sizeof(prompt),
            "You are a specialist assisting the coordinator.\n\n"
            "Original request:\n%.4000s\n\n"
            "Focus on this subproblem:\n%.5000s\n\n"
            "Solve ONLY this subproblem. Do not assume another specialist is correct. "
            "State uncertainty clearly. %s",
            original_problem,task->description,
            code_request
                ? "Return one complete, compile-ready C code block, then at most three short notes. Satisfy every explicit constraint. Check null and empty input, signs, trailing characters, overflow before multiplication, output-pointer safety, and all declarations. Do not add a main unless requested."
                : "Explain the key reasoning and conclusion in plain English, using only the detail needed to support the result.");

        char *response =
            task->responses[task->response_count];

        if (ask_specialist(exe,s,code_request,prompt,response,
                           sizeof(task->responses[0])) == 0) {

            ++task->response_count;
            task->response_specialist_indices[task->response_count - 1] = specialist_index;

            if (specialist_slot + 1 < max_attempts) {
                char review_prompt[ATLAS_MAX_TEXT];
                char review[2048] = {0};
                snprintf(review_prompt, sizeof(review_prompt),
                    "You are the small coordinator reviewing one specialist result.\n"
                    "Decide whether this result fully resolves the subproblem.\n"
                    "Reply exactly COMPLETE if it is sufficiently supported.\n"
                    "Otherwise reply NEED_SPECIALIST and name the unresolved gap.\n\n"
                    "SUBPROBLEM:\n%.5000s\n\nSPECIALIST RESULT:\n%.7000s",
                    task->description, response);
                if (ask(exe, router_model, review_prompt, review, sizeof(review)) == 0 &&
                    answer_meets_basic_code_requirements(original_problem,response) &&
                    contains_ci(review, "COMPLETE") &&
                    !contains_ci(review, "NEED_SPECIALIST"))
                    break;
            }
        }
    }

    return task->response_count > 0 ? 0 : 1;
}

int atlas_reason_staged(
    const char *exe,
    const char *router_model,
    const char *coordinator_model,
    int allow_reference,
    const char *problem,
    AtlasReasoningSession *session)
{
    if (!exe ||
        !router_model ||
        !coordinator_model ||
        !problem ||
        !session)
        return 1;

    memset(session, 0, sizeof(*session));

    safe_copy(
        session->original_problem,
        sizeof(session->original_problem),
        problem
    );

    atlas_conversation_init();

    atlas_conversation_message(
        ATLAS_SPEAKER_ATLAS,
        router_model,
        "Beginning multi-specialist reasoning session."
    );

    /*
     * ---------------------------------------------------------
     * PHASE 1: ATLAS DECOMPOSES THE PROBLEM
     * ---------------------------------------------------------
     */

    char decomposition_prompt[ATLAS_MAX_TEXT];

    snprintf(
        decomposition_prompt,
        sizeof(decomposition_prompt),

        "You are Atlas, the coordinating reasoning system.\n\n"

        "Break the following problem into the smallest useful "
        "independent subproblems that can be delegated to "
        "specialist models.\n\n"

        "Do not solve the problem yet.\n"
        "Identify what must be investigated.\n\n"

        "Return one task per line using exactly this form:\n"
        "TASK 1: description\n"
        "TASK 2: description\n"
        "TASK 3: description\n\n"

        "Original problem:\n%s",

        problem
    );

    char decomposition[ATLAS_MAX_TEXT];

    if (ask(
            exe,
            coordinator_model,
            decomposition_prompt,
            decomposition,
            sizeof(decomposition)) != 0) {

        return 2;
    }

    atlas_conversation_message(
        ATLAS_SPEAKER_ATLAS,
        coordinator_model,
        decomposition
    );

    parse_decomposition(
        decomposition,
        session
    );

    /*
     * ---------------------------------------------------------
     * PHASE 2: ROUTE AND INVESTIGATE
     * ---------------------------------------------------------
     */

    for (int i = 0;
         i < session->task_count;
         ++i) {

        AtlasSubtask *task =
            &session->tasks[i];

        choose_specialists(exe, router_model, task);

        if (collect_specialist_answers(exe, router_model,
                                       session->original_problem, task) != 0)
            return 5;

        for (int r = 0;
             r < task->response_count;
             ++r) {

            int specialist_index = task->response_specialist_indices[r];
            const char *specialist_name =
                specialist_index >= 0 && specialist_index < g_specialist_count
                ? g_specialists[specialist_index].name : "selected specialist";
            atlas_conversation_message(
                ATLAS_SPEAKER_SPECIALIST,
                specialist_name,
                task->responses[r]
            );
        }
    }

    /*
     * ---------------------------------------------------------
     * PHASE 3: ATLAS SYNTHESIS
     * ---------------------------------------------------------
     */

    char synthesis_prompt[ATLAS_MAX_TEXT];

    size_t used = 0;

    if (!append_format(synthesis_prompt, sizeof(synthesis_prompt), &used,
        "You are Atlas.\n\n"
        "You must combine the independent specialist "
        "investigations below into one correct solution.\n\n"

        "Do NOT simply choose the most confident response.\n"
        "Analyze all responses.\n"
        "Preserve useful information from different specialists.\n"
        "Identify contradictions.\n"
        "Resolve contradictions using reasoning.\n"
        "Do not assume agreement means correctness.\n\n"
        "Write in clear natural English. For code requests, include "
        "complete C code that honors every stated constraint, then "
        "briefly explain how to build and use it.\n\n"

        "ORIGINAL PROBLEM:\n%s\n\n", problem))
        used = sizeof(synthesis_prompt) - 1;

    for (int i = 0;
         i < session->task_count &&
         used + 256 < sizeof(synthesis_prompt);
         ++i) {

        AtlasSubtask *task =
            &session->tasks[i];

        if (!append_format(synthesis_prompt, sizeof(synthesis_prompt), &used,
                           "SUBPROBLEM %d:\n%s\n\n", task->id, task->description))
            break;

        for (int r = 0;
             r < task->response_count &&
             used + 256 < sizeof(synthesis_prompt);
             ++r) {

            if (!append_format(synthesis_prompt, sizeof(synthesis_prompt), &used,
                               "SPECIALIST RESPONSE %d:\n%s\n\n", r + 1,
                               task->responses[r]))
                break;
        }
    }

    (void)append_format(synthesis_prompt, sizeof(synthesis_prompt), &used,
        "Now produce the best unified answer.\n"
        "Explain why the combined answer is correct.\n"
        "If important uncertainty remains, state exactly what "
        "is uncertain.");

    if (ask(
            exe,
            coordinator_model,
            synthesis_prompt,
            session->synthesis,
            sizeof(session->synthesis)) != 0 || !session->synthesis[0]) {

        return 3;
    }

    atlas_conversation_message(
        ATLAS_SPEAKER_ATLAS,
        coordinator_model,
        session->synthesis
    );

    /*
     * ---------------------------------------------------------
     * PHASE 4: VERIFY THE SYNTHESIS
     * ---------------------------------------------------------
     */

    char verification_prompt[ATLAS_MAX_TEXT];

    snprintf(
        verification_prompt,
        sizeof(verification_prompt),

        "You are Atlas performing final verification.\n\n"

        "Original problem:\n%s\n\n"

        "Proposed synthesized answer:\n%s\n\n"

        "Check this answer critically.\n"
        "Look specifically for:\n"
        "1. contradictions between specialist findings\n"
        "2. unsupported assumptions\n"
        "3. missing information\n"
        "4. logical errors\n"
        "5. incorrect synthesis of partial answers\n"
        "For a programming answer, check every explicit requirement "
        "one by one and inspect the C declarations and memory handling. "
        "Reject code that cannot compile or omits a requested constraint.\n\n"

        "Return either:\n"
        "VERIFIED: followed by a concise justification\n"
        "or\n"
        "REJECTED: followed by the specific problems.",

        problem,
        session->synthesis
    );

    if (ask(
            exe,
            coordinator_model,
            verification_prompt,
            session->verification,
            sizeof(session->verification)) != 0) {

        return 4;
    }

    atlas_conversation_message(
        ATLAS_SPEAKER_ATLAS,
        coordinator_model,
        session->verification
    );

    session->verified = verification_passes(session->verification);
    if (session->verified && !answer_meets_basic_code_requirements(problem,session->synthesis)) {
        session->verified = 0;
        snprintf(session->verification,sizeof(session->verification),
                 "REJECTED: The proposed implementation is missing source code or one or more explicitly requested C identifiers.");
    }

    session->rounds = 1;

    for (int escalation = 0;
         allow_reference && !session->verified && escalation < 2 &&
         strcmp(coordinator_model, router_model) != 0;
         ++escalation) {
        char reference_prompt[ATLAS_MAX_TEXT];
        size_t reference_used = 0;
        (void)append_format(reference_prompt, sizeof(reference_prompt), &reference_used,
            "You are Atlas's larger reference model. Independently check the proposed answer.\n"
            "Use the specialist findings below. Identify concrete errors and give a corrected answer.\n\n"
            "ORIGINAL PROBLEM:\n%s\n\nCURRENT COORDINATOR DRAFT:\n%s\n\nSMALL MODEL CHECK:\n%s\n\n",
            problem, session->synthesis, session->verification);
        for (int i = 0; i < session->task_count && reference_used + 256 < sizeof(reference_prompt); ++i) {
            AtlasSubtask *task = &session->tasks[i];
            (void)append_format(reference_prompt, sizeof(reference_prompt), &reference_used,
                                "SUBTASK %d: %s\n", task->id, task->description);
            for (int r = 0; r < task->response_count && reference_used + 256 < sizeof(reference_prompt); ++r)
                if (!append_format(reference_prompt, sizeof(reference_prompt), &reference_used,
                                   "SMALL SPECIALIST FINDING:\n%s\n\n", task->responses[r]))
                    break;
        }
        char reference[ATLAS_MAX_TEXT] = {0};
        if (ask(exe, coordinator_model, reference_prompt, reference, sizeof(reference)) == 0) {
            atlas_conversation_message(ATLAS_SPEAKER_TEACHER, coordinator_model, reference);
            char repair_prompt[ATLAS_MAX_TEXT];
            snprintf(repair_prompt, sizeof(repair_prompt),
                "You are Atlas, the coordinator. Produce one corrected answer using the specialist findings and this larger reference.\n\n"
                "Original problem:\n%.3000s\n\nSmall-model draft:\n%.3000s\n\n"
                "Reference analysis:\n%.5000s\n\n"
                "Resolve disagreements carefully. Keep only supported conclusions. State any remaining uncertainty.",
                problem, session->synthesis, reference);
            char repaired[ATLAS_MAX_TEXT] = {0};
            if (ask(exe, coordinator_model, repair_prompt, repaired, sizeof(repaired)) == 0 && repaired[0]) {
                safe_copy(session->synthesis, sizeof(session->synthesis), repaired);
                atlas_conversation_message(ATLAS_SPEAKER_ATLAS, coordinator_model, session->synthesis);
                snprintf(verification_prompt, sizeof(verification_prompt),
                    "Check this final answer against the original problem and specialist findings.\n"
                    "Return exactly VERIFIED: followed by one short reason, or REJECTED: followed by the specific unresolved errors.\n\n"
                    "Original problem:\n%.4500s\n\nFinal answer:\n%.7000s\n\n"
                    "Reference analysis:\n%.3000s", problem, session->synthesis, reference);
                memset(session->verification, 0, sizeof(session->verification));
                if (ask(exe, coordinator_model, verification_prompt, session->verification,
                        sizeof(session->verification)) == 0) {
                    atlas_conversation_message(ATLAS_SPEAKER_ATLAS, coordinator_model,
                                                session->verification);
                    session->verified = verification_passes(session->verification);
                    if (session->verified && !answer_meets_basic_code_requirements(problem,session->synthesis)) {
                        session->verified = 0;
                        snprintf(session->verification,sizeof(session->verification),
                                 "REJECTED: The repaired answer is missing source code or one or more explicitly requested C identifiers.");
                    }
                    ++session->rounds;
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if (asks_for_implementation(problem) && !answer_meets_basic_code_requirements(problem,session->synthesis)) {
        session->verified = 0;
        snprintf(session->verification, sizeof(session->verification),
                 "REJECTED: The answer does not contain source code satisfying the request's basic identifier checks.");
        atlas_conversation_message(ATLAS_SPEAKER_ATLAS, coordinator_model,
                                    session->verification);
    }

    record_session_feedback(session);

    /*
     * If verification fails, the caller can perform another
     * reasoning pass. We deliberately expose this state instead
     * of pretending the answer is correct.
     */

    return 0;
}

int atlas_reason(
    const char *exe,
    const char *router_model,
    const char *problem,
    AtlasReasoningSession *session)
{
    return atlas_reason_staged(exe, router_model, router_model, 0, problem, session);
}
