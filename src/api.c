#include <ctype.h>
#include <float.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "wapiti.h"
#include "gradient.h"
#include "model.h"
#include "quark.h"
#include "reader.h"
#include "sequence.h"
#include "thread.h"
#include "tools.h"
#include "decoder.h"
#include "api.h"
#include "progress.h"
#include "trainers.h"


/* To maintain c99 compatibility.. */
static char *strdup(const char *str)
{
  char *cpy = NULL;
  if (str)
  {
    cpy = malloc(strlen(str)+1);
    if (cpy)
      strcpy(cpy, str);
  }
  return cpy;
}

/* Converts a BIO-formatted string to a raw sequence type */
static raw_t *api_str2raw(char *seq) {
  int size = 32; // Initial number of lines in raw_t
  int cnt = 0;
  char *line;

  raw_t *raw = xmalloc(sizeof(raw_t) + sizeof(char *) * size);

  for (line = strtok(seq, "\n") ; line ; line = strtok(NULL, "\n")) {
    // Make sure there's room and add the line
    if (cnt == size) {
      size *= 1.4;
      raw = xrealloc(raw, sizeof(raw_t) + sizeof(char *) * size);
    }
    raw->lines[cnt++] = strdup(line);
  }
  raw->len = cnt;
  return raw;
}

/*
 * Redeclarations of necessary unexported wapiti data
 */

/* Default training algorithm */
static void trn_auto(mdl_t *mdl) {
	const int maxiter = mdl->opt->maxiter;
	mdl->opt->maxiter = 3;
	trn_sgdl1(mdl);
	mdl->opt->maxiter = maxiter;
	trn_lbfgs(mdl);
}

/* Available model types*/
static const char *typ_lst[] = {
	"maxent",
	"memm",
	"crf"
};
static const uint32_t typ_cnt = sizeof(typ_lst) / sizeof(typ_lst[0]);


/* Available training algorithms*/
static const struct {
	char *name;
	void (* train)(mdl_t *mdl);
} trn_lst[] = {
	{"l-bfgs", trn_lbfgs},
	{"sgd-l1", trn_sgdl1},
	{"bcd",    trn_bcd  },
	{"rprop",  trn_rprop},
	{"rprop+", trn_rprop},
	{"rprop-", trn_rprop},
	{"auto",   trn_auto }
};
static const int trn_cnt = sizeof(trn_lst) / sizeof(trn_lst[0]);


/* Initializes model  */
mdl_t *api_new_model(opt_t *options, char *patterns) {
  mdl_t *mdl = mdl_new(rdr_new(options->maxent));
  mdl->opt = options;

  // Make sure the selected model type is valid
  uint32_t typ;
  for (typ = 0; typ < typ_cnt; typ++)
    if (!strcmp(mdl->opt->type, typ_lst[typ]))
      break;
  if (typ == typ_cnt)
    fatal("unknown model type '%s'", mdl->opt->type);
  mdl->type = typ;

  // Load patterns from a string
  if (patterns != NULL) {
    api_load_patterns(mdl, patterns);
  }

  // Initialize training data
  dat_t *dat = xmalloc(sizeof(dat_t));
  dat->nseq = 0;
  dat->mlen = 0;
  dat->lbl = true;
  dat->seq = NULL;
  mdl->train = dat;

  return mdl;
}

/* Initializes a model and loads data from model file */
mdl_t *api_load_model(char *filename, opt_t *options) {
  mdl_t *mdl = api_new_model(options, NULL);

  FILE *file = fopen(filename, "r");
  if (file == NULL)
    pfatal("cannot open input model file: %s", filename);
  mdl_load(mdl, file);
  return mdl;
}

/*
 * Splits a raw BIO-formatted string into lines, annotates them and
 * returns a copy of input string with an added label column.
 */
char *api_label_seq(mdl_t *mdl, char *lines) {
    size_t outsize = strlen(lines);
	qrk_t *lbls = mdl->reader->lbl;
    raw_t *raw = api_str2raw(lines);
	seq_t *seq = rdr_raw2seq(mdl->reader, raw, mdl->opt->check);
	const int T = seq->len;

	uint32_t *out = xmalloc(sizeof(uint32_t) * T);
	double *psc = xmalloc(sizeof(double) * T);
	double *scs = xmalloc(sizeof(double));

    tag_viterbi(mdl, seq, out, scs, psc);

    // Allocate some intial memory for the output string.
    // Account for the input string + a little extra per line for
    // label and whitespace
    outsize += 5 * T;
    char *lblseq = xmalloc(outsize);
    const char *lblstr;
    size_t rowsize;
    size_t pos = 0;

	// Build the output string
    for (int t = 0; t < T; t++) {
      lblstr = qrk_id2str(lbls, out[t]);
      // Size: input line  + \t + label + \n + \0
      rowsize = strlen(raw->lines[t]) + strlen(lblstr) + 3;
      if (pos+rowsize > outsize) {
        outsize += rowsize*(T-t);
        lblseq = xrealloc(lblseq, outsize);
      }
      pos += snprintf(lblseq+pos, outsize-pos, "%s\t%s\n", raw->lines[t], lblstr);
    }
    // Terminate the output string
    lblseq[pos] = '\0';

	// Free memory used for this sequence
	free(scs);
	free(psc);
	free(out);
	rdr_freeraw(raw);
	rdr_freeseq(seq);
    return lblseq;
}


/* Compiles lines of patterns and stores them in the model */
void api_load_patterns(mdl_t *mdl, char *lines) {
  rdr_t *rdr = mdl->reader;
  for (char *line = strtok(lines, "\n") ; line ; line = strtok(NULL, "\n")) {

    // Remove comments and trailing spaces
    int end = strcspn(line, "#");
    while (end != 0 && isspace(line[end - 1]))
      end--;
    if (end == 0)
      continue;

    line[end] = '\0';
    line[0] = tolower(line[0]);

    // Avoid messing with the original pattern string
    char *patstr = strdup(line);

    // Compile pattern and add it to the list
    pat_t *pat = pat_comp(patstr);
    rdr->npats++;
    switch (line[0]) {
      case 'u': rdr->nuni++; break;
      case 'b': rdr->nbi++; break;
      case '*': rdr->nuni++;
        rdr->nbi++; break;
      default:
        fatal("unknown pattern type '%c'", line[0]);
    }
    rdr->pats = xrealloc(rdr->pats, sizeof(char *) * rdr->npats);
    rdr->pats[rdr->npats - 1] = pat;
    rdr->ntoks = max(rdr->ntoks, pat->ntoks);
  }

}

/* Adds a sequence of BIO-formatted training data to the model. */
void api_add_train_seq(mdl_t *mdl, char *lines) {
  dat_t *dat = mdl->train;
  raw_t *raw = api_str2raw(lines);
  seq_t *seq = rdr_raw2seq(mdl->reader, raw, true);

  // Make room
  dat->seq = xrealloc(dat->seq, sizeof(seq_t *) * (dat->nseq + 1));

  // And store the sequence
  dat->seq[dat->nseq++] = seq;
  dat->mlen = max(dat->mlen, seq->len);
}

/* Trains the model on loaded training data. */
void api_train(mdl_t *mdl) {

  // Get the training method
  int trn;
  for (trn = 0; trn < trn_cnt; trn++)
    if (!strcmp(mdl->opt->algo, trn_lst[trn].name))
      break;
  if (trn == trn_cnt)
    fatal("unknown algorithm '%s'", mdl->opt->algo);

  mdl_sync(mdl);            // Finalize model structure for training
  uit_setup(mdl);           // Setup signal handling to abort training
  trn_lst[trn].train(mdl);
  uit_cleanup(mdl);

}

/* Saves the model to a file. */
void api_save_model(mdl_t *mdl, FILE *file) {
  mdl_save(mdl, file);
}

/* Frees all memory used by the model. */
void api_free_model(mdl_t *mdl) {
  mdl_free(mdl);
}


/* Silly tricks to wrap wapiti's logging/error calls.
 *
 * The Makefile uses the --wrap linker flag to route wapiti's logging
 * and error function calls to the corresponding __wrap_-fuctions.
 *
 * To make it easier to hook in custom logging functions, the wrapped
 * functions in turn call the corresponding functions in the logs
 * array with the log messag as a string. To customize the
 * info-logger, for example, simply assign logs[INFO] a pointer to
 * your logging function.
 *
 */

/* These are the default logging functions */
void inf_log(char *msg) {
  fprintf(stdout, "%s", msg);
}
void wrn_log(char *msg) {
  fprintf(stderr, "wrn: %s\n", msg);
}
void err_log(char *msg) {
  fprintf(stderr, "err: %s\n", msg);
  exit(EXIT_FAILURE);
}

/*
 * The 4 logging levels. FATAL and PFATAL are actually more like final
 * words and and not logging. See wapiti src for more info.
 */
enum loglvl {
  FATAL,
  PFATAL,
  WARNING,
  INFO
};

/* Customize by replacing with logging function pointers */
void (* api_logs[4])(char *msg) = {
  err_log,
  err_log,
  wrn_log,
  inf_log
};

/*
 * Too avoid bloating this with clever allocation logic, log messages
 * are truncated to 1400 chars. Look, it's 10 tweets!
 */
const int MAXLOGMSG = 1400;
char *OOMMSG = "out of memory\n";

/*
 * After fatal log messages the program state should be considered
 * unknown and no resources are freed. The only reason these wrappers
 * don't call exit is to allow custom error handlers to quit on their
 * own terms.
 */
void __wrap_fatal(const char *msg, ...) {
  va_list args;
  char *message = malloc(MAXLOGMSG);
  if (message == NULL) {
    message = OOMMSG;
  } else {
    va_start(args, msg);
    vsnprintf(message, MAXLOGMSG, msg, args);
    va_end(args);
  }
  api_logs[FATAL](message);
}
void __wrap_pfatal(const char *msg, ...) {
  va_list args;
  const char *err = strerror(errno);
  char *message = malloc(MAXLOGMSG + strlen(err) + 4);
  if (message == NULL) {
    message = OOMMSG;
  } else {
    va_start(args, msg);
    vsnprintf(message, MAXLOGMSG, msg, args);
    va_end(args);
    size_t msglen = strlen(message);
	snprintf(message+msglen, MAXLOGMSG-msglen, " <%s>", err);
  }
  api_logs[PFATAL](message);
}

/*
 * Non-fatal log functions don't need to stop execution, and the error
 * message will be freed after the logs-call
 */
void __wrap_warning(const char *msg, ...) {
  va_list args;
  char *message = malloc(MAXLOGMSG);
  if (message == NULL) {
    message = OOMMSG;
  } else {
    va_start(args, msg);
    vsnprintf(message, MAXLOGMSG, msg, args);
    va_end(args);
  }
  api_logs[WARNING](message);
  free(message);
}
void __wrap_info(const char *msg, ...) {
  va_list args;
  char *message = malloc(MAXLOGMSG);
  if (message == NULL) {
    message = OOMMSG;
  } else {
    va_start(args, msg);
    vsnprintf(message, MAXLOGMSG, msg, args);
    va_end(args);
  }
  api_logs[INFO](message);
  free(message);
}

